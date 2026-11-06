package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Tensor is a tensor to be written by Write.
type Tensor struct {
	Name  string
	Shape []uint64
	Type  Type
	Data  []byte
}

// Write writes a GGUF v3 file. Tensor data is aligned to general.alignment if
// present in metadata, otherwise to 32 bytes.
func Write(w io.Writer, metadata []KV, tensors []Tensor) error {
	align := uint64(defaultAlignment)
	for _, kv := range metadata {
		if kv.Key != "general.alignment" {
			continue
		}
		a, ok := kv.Value.(uint32)
		if !ok || a == 0 || a&(a-1) != 0 {
			return fmt.Errorf("gguf: general.alignment must be a power of two uint32")
		}
		align = uint64(a)
	}

	offsets := make([]uint64, len(tensors))
	var off uint64
	for i, t := range tensors {
		if len(t.Shape) > maxDims {
			return fmt.Errorf("gguf: tensor %q has %d dimensions", t.Name, len(t.Shape))
		}
		ti, ok := typeInfo[t.Type]
		if !ok {
			return fmt.Errorf("%w: tensor %q: %v", ErrUnsupported, t.Name, t.Type)
		}
		if len(t.Shape) > 0 && t.Shape[0]%ti.block != 0 {
			return fmt.Errorf("gguf: tensor %q: row size %d is not a multiple of %d", t.Name, t.Shape[0], ti.block)
		}
		info := TensorInfo{Shape: t.Shape, Type: t.Type}
		if uint64(len(t.Data)) != info.Size() {
			return fmt.Errorf("gguf: tensor %q has %d bytes, want %d", t.Name, len(t.Data), info.Size())
		}
		offsets[i] = off
		off += uint64(len(t.Data))
		off += pad(off, align)
	}

	e := &encoder{w: bufio.NewWriter(w)}
	e.u32(magic)
	e.u32(3)
	e.u64(uint64(len(tensors)))
	e.u64(uint64(len(metadata)))
	for _, kv := range metadata {
		e.string(kv.Key)
		if err := e.value(kv.Value, true); err != nil {
			return fmt.Errorf("gguf: key %q: %w", kv.Key, err)
		}
	}
	for i, t := range tensors {
		e.string(t.Name)
		e.u32(uint32(len(t.Shape)))
		for _, d := range t.Shape {
			e.u64(d)
		}
		e.u32(uint32(t.Type))
		e.u64(offsets[i])
	}
	e.zeros(pad(e.n, align))
	for _, t := range tensors {
		e.write(t.Data)
		e.zeros(pad(e.n, align))
	}
	if e.err != nil {
		return e.err
	}
	return e.w.Flush()
}

type encoder struct {
	w   *bufio.Writer
	n   uint64
	err error
	buf [8]byte
}

func (e *encoder) write(p []byte) {
	if e.err != nil {
		return
	}
	var n int
	n, e.err = e.w.Write(p)
	e.n += uint64(n)
}

// zeros writes n zero bytes.
func (e *encoder) zeros(n uint64) {
	var z [256]byte
	for n > 0 {
		k := min(n, uint64(len(z)))
		e.write(z[:k])
		n -= k
	}
}

func (e *encoder) u8(v uint8) { e.write([]byte{v}) }

func (e *encoder) u16(v uint16) {
	binary.LittleEndian.PutUint16(e.buf[:], v)
	e.write(e.buf[:2])
}

func (e *encoder) u32(v uint32) {
	binary.LittleEndian.PutUint32(e.buf[:], v)
	e.write(e.buf[:4])
}

func (e *encoder) u64(v uint64) {
	binary.LittleEndian.PutUint64(e.buf[:], v)
	e.write(e.buf[:8])
}

func (e *encoder) string(s string) {
	e.u64(uint64(len(s)))
	e.write([]byte(s))
}

func (e *encoder) bool(b bool) {
	if b {
		e.u8(1)
	} else {
		e.u8(0)
	}
}

// value encodes v. If tagged, the value type is written first.
func (e *encoder) value(v any, tagged bool) error {
	t, ok := valueType(v)
	if !ok {
		return fmt.Errorf("unsupported value type %T", v)
	}
	if tagged {
		e.u32(t)
	}
	switch v := v.(type) {
	case uint8:
		e.u8(v)
	case int8:
		e.u8(uint8(v))
	case uint16:
		e.u16(v)
	case int16:
		e.u16(uint16(v))
	case uint32:
		e.u32(v)
	case int32:
		e.u32(uint32(v))
	case float32:
		e.u32(math.Float32bits(v))
	case bool:
		e.bool(v)
	case string:
		e.string(v)
	case uint64:
		e.u64(v)
	case int64:
		e.u64(uint64(v))
	case float64:
		e.u64(math.Float64bits(v))
	case []uint8:
		writeArray(e, typeUint8, v, e.u8)
	case []int8:
		writeArray(e, typeInt8, v, func(x int8) { e.u8(uint8(x)) })
	case []uint16:
		writeArray(e, typeUint16, v, e.u16)
	case []int16:
		writeArray(e, typeInt16, v, func(x int16) { e.u16(uint16(x)) })
	case []uint32:
		writeArray(e, typeUint32, v, e.u32)
	case []int32:
		writeArray(e, typeInt32, v, func(x int32) { e.u32(uint32(x)) })
	case []float32:
		writeArray(e, typeFloat32, v, func(x float32) { e.u32(math.Float32bits(x)) })
	case []bool:
		writeArray(e, typeBool, v, e.bool)
	case []string:
		writeArray(e, typeString, v, e.string)
	case []uint64:
		writeArray(e, typeUint64, v, e.u64)
	case []int64:
		writeArray(e, typeInt64, v, func(x int64) { e.u64(uint64(x)) })
	case []float64:
		writeArray(e, typeFloat64, v, func(x float64) { e.u64(math.Float64bits(x)) })
	case []any:
		e.u32(typeArray)
		e.u64(uint64(len(v)))
		for _, x := range v {
			if t, _ := valueType(x); t != typeArray {
				return fmt.Errorf("nested array element has type %T", x)
			}
			if err := e.value(x, false); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeArray[T any](e *encoder, t uint32, s []T, put func(T)) {
	e.u32(t)
	e.u64(uint64(len(s)))
	for _, x := range s {
		put(x)
	}
}

func valueType(v any) (uint32, bool) {
	switch v.(type) {
	case uint8:
		return typeUint8, true
	case int8:
		return typeInt8, true
	case uint16:
		return typeUint16, true
	case int16:
		return typeInt16, true
	case uint32:
		return typeUint32, true
	case int32:
		return typeInt32, true
	case float32:
		return typeFloat32, true
	case bool:
		return typeBool, true
	case string:
		return typeString, true
	case uint64:
		return typeUint64, true
	case int64:
		return typeInt64, true
	case float64:
		return typeFloat64, true
	case []uint8, []int8, []uint16, []int16, []uint32, []int32, []float32,
		[]bool, []string, []uint64, []int64, []float64, []any:
		return typeArray, true
	}
	return 0, false
}
