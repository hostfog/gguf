// Package gguf reads and writes GGUF files, the model format used by
// llama.cpp.
//
// Only the header is parsed when a file is opened; tensor data is read on
// demand through TensorReader.
package gguf

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

var (
	ErrNotGGUF     = errors.New("gguf: not a GGUF file")
	ErrVersion     = errors.New("gguf: unsupported version")
	ErrBigEndian   = errors.New("gguf: big-endian files are not supported")
	ErrMalformed   = errors.New("gguf: malformed file")
	ErrUnsupported = errors.New("gguf: unsupported tensor type")
)

const (
	magic            = 0x46554747 // "GGUF" read as little-endian
	defaultAlignment = 32
	maxDims          = 4
)

// File is a parsed GGUF file.
type File struct {
	Version  uint32
	Metadata []KV
	Tensors  []TensorInfo

	// Alignment is the alignment of tensor data, from general.alignment.
	Alignment uint64

	// DataOffset is the absolute offset of the tensor data section.
	DataOffset int64

	r       io.ReaderAt
	closer  io.Closer
	kv      map[string]int
	tensors map[string]int
}

// TensorInfo describes one tensor. Offset is relative to the file's
// DataOffset.
type TensorInfo struct {
	Name   string
	Shape  []uint64
	Type   Type
	Offset uint64
}

// Elements returns the number of elements in the tensor.
func (t TensorInfo) Elements() uint64 {
	n := uint64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Size returns the size of the tensor data in bytes.
func (t TensorInfo) Size() uint64 {
	info := typeInfo[t.Type]
	if info.block == 0 {
		return 0
	}
	return t.Elements() / info.block * info.bytes
}

// Open opens and parses the GGUF file at path. The caller must Close it.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	gf, err := Read(f, st.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	gf.closer = f
	return gf, nil
}

// Close closes the underlying file if it was opened with Open.
func (f *File) Close() error {
	if f.closer == nil {
		return nil
	}
	return f.closer.Close()
}

// Read parses a GGUF file of the given size from r. r must remain valid for
// as long as tensor data is read from the returned File.
func Read(r io.ReaderAt, size int64) (*File, error) {
	if size < 0 {
		return nil, fmt.Errorf("%w: negative size", ErrMalformed)
	}
	d := newDecoder(r, size)

	if d.u32() != magic {
		if d.err != nil && !errors.Is(d.err, ErrMalformed) {
			return nil, d.err
		}
		return nil, ErrNotGGUF
	}
	f := &File{
		r:         r,
		Version:   d.u32(),
		Alignment: defaultAlignment,
		kv:        make(map[string]int),
		tensors:   make(map[string]int),
	}
	if d.err != nil {
		return nil, d.err
	}
	switch v := f.Version; {
	case v == 2 || v == 3:
	case v == 2<<24 || v == 3<<24:
		return nil, ErrBigEndian
	default:
		return nil, fmt.Errorf("%w %d", ErrVersion, v)
	}

	nTensors := d.u64()
	nKV := d.u64()
	if d.err != nil {
		return nil, d.err
	}
	if err := f.readMetadata(d, nKV); err != nil {
		return nil, err
	}
	if err := f.readTensors(d, nTensors); err != nil {
		return nil, err
	}

	f.DataOffset = d.off + int64(pad(uint64(d.off), f.Alignment))
	for _, t := range f.Tensors {
		if f.DataOffset > size {
			return nil, fmt.Errorf("%w: tensor data out of range", ErrMalformed)
		}
		avail := uint64(size - f.DataOffset)
		if t.Offset > avail || t.Size() > avail-t.Offset {
			return nil, fmt.Errorf("%w: tensor %q data out of range", ErrMalformed, t.Name)
		}
	}
	return f, nil
}

func (f *File) readMetadata(d *decoder, n uint64) error {
	// Each entry needs at least a key length and a value type.
	if n > uint64(d.remaining())/12 {
		return fmt.Errorf("%w: metadata count %d exceeds file size", ErrMalformed, n)
	}
	f.Metadata = make([]KV, 0, min(n, maxPrealloc))
	for i := uint64(0); i < n; i++ {
		key := d.string()
		val := d.value(d.u32(), 0)
		if d.err != nil {
			return d.err
		}
		if _, dup := f.kv[key]; dup {
			return fmt.Errorf("%w: duplicate key %q", ErrMalformed, key)
		}
		f.kv[key] = len(f.Metadata)
		f.Metadata = append(f.Metadata, KV{Key: key, Value: val})
	}

	if v, ok := f.Value("general.alignment"); ok {
		a, ok := v.(uint32)
		if !ok || a == 0 || a&(a-1) != 0 {
			return fmt.Errorf("%w: general.alignment must be a power of two uint32", ErrMalformed)
		}
		f.Alignment = uint64(a)
	}
	return nil
}

func (f *File) readTensors(d *decoder, n uint64) error {
	// name length + dims count + type + offset
	if n > uint64(d.remaining())/24 {
		return fmt.Errorf("%w: tensor count %d exceeds file size", ErrMalformed, n)
	}
	f.Tensors = make([]TensorInfo, 0, min(n, maxPrealloc))
	for i := uint64(0); i < n; i++ {
		t := TensorInfo{Name: d.string()}
		dims := d.u32()
		if d.err == nil && dims > maxDims {
			return fmt.Errorf("%w: tensor %q has %d dimensions", ErrMalformed, t.Name, dims)
		}
		t.Shape = make([]uint64, dims)
		for j := range t.Shape {
			t.Shape[j] = d.u64()
		}
		t.Type = Type(d.u32())
		t.Offset = d.u64()
		if d.err != nil {
			return d.err
		}
		if err := f.checkTensor(t); err != nil {
			return err
		}
		f.tensors[t.Name] = len(f.Tensors)
		f.Tensors = append(f.Tensors, t)
	}
	return nil
}

func (f *File) checkTensor(t TensorInfo) error {
	if _, dup := f.tensors[t.Name]; dup {
		return fmt.Errorf("%w: duplicate tensor %q", ErrMalformed, t.Name)
	}
	info, ok := typeInfo[t.Type]
	if !ok {
		return fmt.Errorf("%w: tensor %q: %v", ErrUnsupported, t.Name, t.Type)
	}
	limit := uint64(math.MaxInt64) / info.bytes * info.block
	n := uint64(1)
	for _, d := range t.Shape {
		if d != 0 && n > limit/d {
			return fmt.Errorf("%w: tensor %q is too large", ErrMalformed, t.Name)
		}
		n *= d
	}
	if len(t.Shape) > 0 && t.Shape[0]%info.block != 0 {
		return fmt.Errorf("%w: tensor %q: row size %d is not a multiple of %v block size %d",
			ErrMalformed, t.Name, t.Shape[0], t.Type, info.block)
	}
	if t.Offset%f.Alignment != 0 {
		return fmt.Errorf("%w: tensor %q offset %d is not aligned", ErrMalformed, t.Name, t.Offset)
	}
	return nil
}

// Lookup returns the tensor with the given name.
func (f *File) Lookup(name string) (TensorInfo, bool) {
	i, ok := f.tensors[name]
	if !ok {
		return TensorInfo{}, false
	}
	return f.Tensors[i], true
}

// TensorReader returns a reader over the raw data of the named tensor.
func (f *File) TensorReader(name string) (*io.SectionReader, error) {
	t, ok := f.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	return io.NewSectionReader(f.r, f.DataOffset+int64(t.Offset), int64(t.Size())), nil
}

// Float32s reads the named tensor and dequantizes it to float32. See
// Dequantize for the supported types.
func (f *File) Float32s(name string) ([]float32, error) {
	t, ok := f.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	if _, ok := dequantizers[t.Type]; !ok {
		return nil, fmt.Errorf("%w: %v", ErrUnsupported, t.Type)
	}
	sr, err := f.TensorReader(name)
	if err != nil {
		return nil, err
	}
	data := make([]byte, sr.Size())
	if _, err := io.ReadFull(sr, data); err != nil {
		return nil, err
	}
	return Dequantize(t.Type, data)
}

// ParameterCount returns the total number of elements across all tensors.
func (f *File) ParameterCount() uint64 {
	var n uint64
	for _, t := range f.Tensors {
		n += t.Elements()
	}
	return n
}

func pad(off, align uint64) uint64 {
	return (align - off%align) % align
}
