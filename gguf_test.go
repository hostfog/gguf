package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func build(t testing.TB, md []KV, tensors []Tensor) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := Write(&buf, md, tensors); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func parse(b []byte) (*File, error) {
	return Read(bytes.NewReader(b), int64(len(b)))
}

func TestRoundTrip(t *testing.T) {
	md := []KV{
		{"general.architecture", "llama"},
		{"general.name", "tiny"},
		{"general.file_type", uint32(7)},
		{"llama.context_length", uint32(4096)},
		{"u8", uint8(1)},
		{"i8", int8(-2)},
		{"u16", uint16(3)},
		{"i16", int16(-4)},
		{"i32", int32(-5)},
		{"u64", uint64(1 << 40)},
		{"i64", int64(-1 << 40)},
		{"f32", float32(0.5)},
		{"f64", 1.25},
		{"bool", true},
		{"empty", ""},
		{"tokens", []string{"<s>", "</s>", "hello"}},
		{"scores", []float32{0, -1.5}},
		{"bytes", []uint8{}},
		{"nested", []any{[]int32{1, 2}, []string{"a"}, []any{[]bool{true}}}},
	}
	tensors := []Tensor{
		{Name: "a", Shape: []uint64{3}, Type: TypeF32, Data: bytes.Repeat([]byte{1}, 12)},
		{Name: "b", Shape: []uint64{32, 2}, Type: TypeQ8_0, Data: bytes.Repeat([]byte{2}, 68)},
		{Name: "c", Shape: []uint64{0}, Type: TypeF16},
	}

	f, err := parse(build(t, md, tensors))
	if err != nil {
		t.Fatal(err)
	}
	if f.Version != 3 || f.Alignment != 32 {
		t.Errorf("version %d alignment %d", f.Version, f.Alignment)
	}
	if !reflect.DeepEqual(f.Metadata, md) {
		t.Errorf("metadata mismatch:\n got %#v\nwant %#v", f.Metadata, md)
	}
	if f.Architecture() != "llama" || f.Name() != "tiny" || f.ContextLength() != 4096 {
		t.Errorf("Architecture/Name/ContextLength = %q %q %d", f.Architecture(), f.Name(), f.ContextLength())
	}
	if ft, ok := f.FileType(); !ok || ft.String() != "Q8_0" {
		t.Errorf("FileType = %v, %v", ft, ok)
	}
	if got := f.ParameterCount(); got != 67 {
		t.Errorf("ParameterCount = %d, want 67", got)
	}
	if f.DataOffset%32 != 0 {
		t.Errorf("DataOffset %d not aligned", f.DataOffset)
	}

	for _, want := range tensors {
		info, ok := f.Lookup(want.Name)
		if !ok {
			t.Fatalf("tensor %q missing", want.Name)
		}
		if info.Offset%32 != 0 || info.Type != want.Type || !reflect.DeepEqual(info.Shape, want.Shape) {
			t.Errorf("tensor %q = %+v", want.Name, info)
		}
		sr, err := f.TensorReader(want.Name)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := io.ReadAll(sr)
		if !bytes.Equal(got, want.Data) {
			t.Errorf("tensor %q data = %v, want %v", want.Name, got, want.Data)
		}
	}
	if _, err := f.TensorReader("missing"); err == nil {
		t.Error("expected error for missing tensor")
	}
}

func TestAccessors(t *testing.T) {
	f, err := parse(build(t, []KV{
		{"s", "x"},
		{"u8", uint8(7)},
		{"i8", int8(-7)},
		{"u32", uint32(7)},
		{"big", uint64(math.MaxUint64)},
		{"f32", float32(1.5)},
		{"b", true},
		{"list", []string{"a", "b"}},
	}, nil))
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		key  string
		call func(string) (any, bool)
		want any
		ok   bool
	}{
		{"s", wrap(f.String), "x", true},
		{"u8", wrap(f.String), "", false},
		{"missing", wrap(f.String), "", false},
		{"u8", wrap(f.Uint), uint64(7), true},
		{"u32", wrap(f.Uint), uint64(7), true},
		{"i8", wrap(f.Uint), uint64(0), false},
		{"big", wrap(f.Uint), uint64(math.MaxUint64), true},
		{"s", wrap(f.Uint), uint64(0), false},
		{"i8", wrap(f.Int), int64(-7), true},
		{"u32", wrap(f.Int), int64(7), true},
		{"big", wrap(f.Int), int64(0), false},
		{"f32", wrap(f.Float), 1.5, true},
		{"u8", wrap(f.Float), 0.0, false},
		{"b", wrap(f.Bool), true, true},
		{"s", wrap(f.Bool), false, false},
		{"list", wrap(f.Strings), []string{"a", "b"}, true},
		{"s", wrap(f.Strings), []string(nil), false},
	}
	for _, tt := range tests {
		got, ok := tt.call(tt.key)
		if ok != tt.ok || !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%s: got %v, %v; want %v, %v", tt.key, got, ok, tt.want, tt.ok)
		}
	}
}

func wrap[T any](fn func(string) (T, bool)) func(string) (any, bool) {
	return func(k string) (any, bool) { return fn(k) }
}

func TestAlignment(t *testing.T) {
	md := []KV{{"general.alignment", uint32(64)}}
	tensors := []Tensor{
		{Name: "a", Shape: []uint64{1}, Type: TypeF32, Data: make([]byte, 4)},
		{Name: "b", Shape: []uint64{1}, Type: TypeF32, Data: make([]byte, 4)},
	}
	f, err := parse(build(t, md, tensors))
	if err != nil {
		t.Fatal(err)
	}
	if f.Alignment != 64 || f.DataOffset%64 != 0 || f.Tensors[1].Offset != 64 {
		t.Errorf("alignment %d, data offset %d, second tensor at %d", f.Alignment, f.DataOffset, f.Tensors[1].Offset)
	}

	if err := Write(io.Discard, []KV{{"general.alignment", uint32(3)}}, nil); err == nil {
		t.Error("expected error for non power of two alignment")
	}
}

func TestTypeSizes(t *testing.T) {
	tests := []struct {
		typ          Type
		block, bytes int
	}{
		{TypeF32, 1, 4},
		{TypeBF16, 1, 2},
		{TypeQ4_0, 32, 18},
		{TypeQ8_0, 32, 34},
		{TypeQ2_K, 256, 84},
		{TypeQ3_K, 256, 110},
		{TypeQ4_K, 256, 144},
		{TypeQ5_K, 256, 176},
		{TypeQ6_K, 256, 210},
		{TypeQ8_K, 256, 292},
		{TypeIQ2_XXS, 256, 66},
		{TypeIQ4_XS, 256, 136},
		{TypeIQ1_M, 256, 56},
		{TypeMXFP4, 32, 17},
		{Type(4), 0, 0},
	}
	for _, tt := range tests {
		if tt.typ.BlockSize() != tt.block || tt.typ.BlockBytes() != tt.bytes {
			t.Errorf("%v: block %d bytes %d, want %d %d", tt.typ, tt.typ.BlockSize(), tt.typ.BlockBytes(), tt.block, tt.bytes)
		}
	}

	q4k := TensorInfo{Shape: []uint64{4096, 4096}, Type: TypeQ4_K}
	if got := q4k.Size(); got != 4096*4096/256*144 {
		t.Errorf("Q4_K size = %d", got)
	}
}

// raw builds a GGUF header by hand so tests can produce malformed files.
type raw struct{ bytes.Buffer }

func (r *raw) u32(v uint32) *raw { binary.Write(r, binary.LittleEndian, v); return r }
func (r *raw) u64(v uint64) *raw { binary.Write(r, binary.LittleEndian, v); return r }
func (r *raw) str(s string) *raw { r.u64(uint64(len(s))); r.WriteString(s); return r }

func header(nTensors, nKV uint64) *raw {
	r := &raw{}
	return r.u32(magic).u32(3).u64(nTensors).u64(nKV)
}

func TestReadErrors(t *testing.T) {
	valid := build(t, []KV{{"k", "v"}}, []Tensor{{Name: "t", Shape: []uint64{4}, Type: TypeF32, Data: make([]byte, 16)}})

	tests := []struct {
		name string
		data []byte
		err  error
	}{
		{"empty", nil, ErrNotGGUF},
		{"wrong magic", []byte("GGML\x03\x00\x00\x00"), ErrNotGGUF},
		{"version 1", (&raw{}).u32(magic).u32(1).Bytes(), ErrVersion},
		{"big endian", (&raw{}).u32(magic).u32(3 << 24).Bytes(), ErrBigEndian},
		{"truncated", valid[:30], ErrMalformed},
		{"truncated counts", valid[:12], ErrMalformed},
		{"truncated data", valid[:len(valid)-17], ErrMalformed},
		{"huge kv count", header(0, math.MaxUint64).Bytes(), ErrMalformed},
		{"huge tensor count", header(math.MaxUint64, 0).Bytes(), ErrMalformed},
		{"huge string", header(0, 1).u64(math.MaxUint64).Bytes(), ErrMalformed},
		{"huge array", header(0, 1).str("k").u32(typeArray).u32(typeUint64).u64(1 << 60).Bytes(), ErrMalformed},
		{"unknown value type", header(0, 1).str("k").u32(99).Bytes(), ErrMalformed},
		{"unknown array type", header(0, 1).str("k").u32(typeArray).u32(99).u64(0).Bytes(), ErrMalformed},
		{"duplicate key", header(0, 2).str("k").u32(typeUint32).u32(1).str("k").u32(typeUint32).u32(1).Bytes(), ErrMalformed},
		{"bad alignment", header(0, 1).str("general.alignment").u32(typeUint32).u32(3).Bytes(), ErrMalformed},
		{"alignment type", header(0, 1).str("general.alignment").u32(typeUint64).u64(32).Bytes(), ErrMalformed},
		{"too many dims", header(1, 0).str("t").u32(5).Bytes(), ErrMalformed},
		{"unknown tensor type", header(1, 0).str("t").u32(1).u64(1).u32(4).u64(0).Bytes(), ErrUnsupported},
		{"misaligned offset", header(1, 0).str("t").u32(1).u64(1).u32(0).u64(4).Bytes(), ErrMalformed},
		{"partial block", header(1, 0).str("t").u32(1).u64(16).u32(uint32(TypeQ4_0)).u64(0).Bytes(), ErrMalformed},
		{"overflowing shape", header(1, 0).str("t").u32(2).u64(1 << 32).u64(1 << 32).u32(0).u64(0).Bytes(), ErrMalformed},
		{"data out of range", header(1, 0).str("t").u32(1).u64(1 << 20).u32(0).u64(0).Bytes(), ErrMalformed},
		{"duplicate tensor", header(2, 0).str("t").u32(1).u64(0).u32(0).u64(0).str("t").u32(1).u64(0).u32(0).u64(0).Bytes(), ErrMalformed},
	}

	deep := header(0, 1).str("k")
	for range maxDepth + 1 {
		deep.u32(typeArray).u32(typeArray).u64(1)
	}
	tests = append(tests, struct {
		name string
		data []byte
		err  error
	}{"nested too deep", deep.Bytes(), ErrMalformed})

	for _, tt := range tests {
		_, err := parse(tt.data)
		if !errors.Is(err, tt.err) {
			t.Errorf("%s: err = %v, want %v", tt.name, err, tt.err)
		}
	}
}

func TestVersion2(t *testing.T) {
	b := build(t, []KV{{"k", "v"}}, nil)
	binary.LittleEndian.PutUint32(b[4:], 2)
	f, err := parse(b)
	if err != nil || f.Version != 2 {
		t.Fatalf("v2: %v", err)
	}
}

func TestWriteErrors(t *testing.T) {
	tests := []struct {
		name    string
		md      []KV
		tensors []Tensor
	}{
		{"unsupported value", []KV{{"k", 1}}, nil},
		{"bad nested array", []KV{{"k", []any{"x"}}}, nil},
		{"unknown type", nil, []Tensor{{Name: "t", Shape: []uint64{1}, Type: Type(4)}}},
		{"wrong data size", nil, []Tensor{{Name: "t", Shape: []uint64{2}, Type: TypeF32, Data: make([]byte, 4)}}},
		{"partial block", nil, []Tensor{{Name: "t", Shape: []uint64{16}, Type: TypeQ4_0, Data: make([]byte, 9)}}},
		{"too many dims", nil, []Tensor{{Name: "t", Shape: []uint64{1, 1, 1, 1, 1}, Type: TypeF32, Data: make([]byte, 4)}}},
	}
	for _, tt := range tests {
		if err := Write(io.Discard, tt.md, tt.tensors); err == nil {
			t.Errorf("%s: expected error", tt.name)
		}
	}
}

func TestOpen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	data := []byte{0, 0, 0x80, 0x3f} // 1.0
	if err := os.WriteFile(path, build(t, nil, []Tensor{{Name: "w", Shape: []uint64{1}, Type: TypeF32, Data: data}}), 0o644); err != nil {
		t.Fatal(err)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	got, err := f.Float32s("w")
	if err != nil || len(got) != 1 || got[0] != 1 {
		t.Fatalf("Float32s = %v, %v", got, err)
	}

	if _, err := Open(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Error("expected error opening missing file")
	}
}
