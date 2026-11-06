package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	typeUint8   = 0
	typeInt8    = 1
	typeUint16  = 2
	typeInt16   = 3
	typeUint32  = 4
	typeInt32   = 5
	typeFloat32 = 6
	typeBool    = 7
	typeString  = 8
	typeArray   = 9
	typeUint64  = 10
	typeInt64   = 11
	typeFloat64 = 12
)

const (
	maxPrealloc = 1 << 16
	maxDepth    = 8
)

// elemSize is the smallest encoded size of a value of type t, or 0 if t is
// unknown.
func elemSize(t uint32) uint64 {
	switch t {
	case typeUint8, typeInt8, typeBool:
		return 1
	case typeUint16, typeInt16:
		return 2
	case typeUint32, typeInt32, typeFloat32:
		return 4
	case typeUint64, typeInt64, typeFloat64, typeString:
		return 8
	case typeArray:
		return 12
	}
	return 0
}

// decoder reads little-endian GGUF values. The first error is sticky: once
// set, all reads return zero values.
type decoder struct {
	r    *bufio.Reader
	off  int64
	size int64
	err  error
	buf  [8]byte
}

func newDecoder(r io.ReaderAt, size int64) *decoder {
	return &decoder{r: bufio.NewReaderSize(io.NewSectionReader(r, 0, size), 1<<16), size: size}
}

func (d *decoder) remaining() int64 { return d.size - d.off }

func (d *decoder) fail(format string, args ...any) {
	if d.err == nil {
		d.err = fmt.Errorf("%w: "+format, append([]any{ErrMalformed}, args...)...)
	}
}

func (d *decoder) read(p []byte) []byte {
	if d.err != nil {
		clear(p)
		return p
	}
	if int64(len(p)) > d.remaining() {
		d.fail("unexpected end of file at offset %d", d.off)
		clear(p)
		return p
	}
	if _, err := io.ReadFull(d.r, p); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			d.fail("unexpected end of file at offset %d", d.off)
		} else {
			d.err = err
		}
		clear(p)
		return p
	}
	d.off += int64(len(p))
	return p
}

func (d *decoder) u8() uint8   { return d.read(d.buf[:1])[0] }
func (d *decoder) u16() uint16 { return binary.LittleEndian.Uint16(d.read(d.buf[:2])) }
func (d *decoder) u32() uint32 { return binary.LittleEndian.Uint32(d.read(d.buf[:4])) }
func (d *decoder) u64() uint64 { return binary.LittleEndian.Uint64(d.read(d.buf[:8])) }

func (d *decoder) string() string {
	n := d.u64()
	if d.err != nil {
		return ""
	}
	if n > uint64(d.remaining()) {
		d.fail("string length %d exceeds file size", n)
		return ""
	}
	return string(d.read(make([]byte, n)))
}

func (d *decoder) value(t uint32, depth int) any {
	switch t {
	case typeUint8:
		return d.u8()
	case typeInt8:
		return int8(d.u8())
	case typeUint16:
		return d.u16()
	case typeInt16:
		return int16(d.u16())
	case typeUint32:
		return d.u32()
	case typeInt32:
		return int32(d.u32())
	case typeFloat32:
		return math.Float32frombits(d.u32())
	case typeBool:
		return d.u8() != 0
	case typeString:
		return d.string()
	case typeUint64:
		return d.u64()
	case typeInt64:
		return int64(d.u64())
	case typeFloat64:
		return math.Float64frombits(d.u64())
	case typeArray:
		return d.array(depth)
	}
	d.fail("unknown value type %d", t)
	return nil
}

func (d *decoder) array(depth int) any {
	if depth >= maxDepth {
		d.fail("arrays nested too deeply")
		return nil
	}
	t := d.u32()
	n := d.u64()
	if d.err != nil {
		return nil
	}
	size := elemSize(t)
	if size == 0 {
		d.fail("unknown array element type %d", t)
		return nil
	}
	if n > uint64(d.remaining())/size {
		d.fail("array length %d exceeds file size", n)
		return nil
	}

	switch t {
	case typeUint8:
		return readN(d, n, d.u8)
	case typeInt8:
		return readN(d, n, func() int8 { return int8(d.u8()) })
	case typeUint16:
		return readN(d, n, d.u16)
	case typeInt16:
		return readN(d, n, func() int16 { return int16(d.u16()) })
	case typeUint32:
		return readN(d, n, d.u32)
	case typeInt32:
		return readN(d, n, func() int32 { return int32(d.u32()) })
	case typeFloat32:
		return readN(d, n, func() float32 { return math.Float32frombits(d.u32()) })
	case typeBool:
		return readN(d, n, func() bool { return d.u8() != 0 })
	case typeString:
		return readN(d, n, d.string)
	case typeUint64:
		return readN(d, n, d.u64)
	case typeInt64:
		return readN(d, n, func() int64 { return int64(d.u64()) })
	case typeFloat64:
		return readN(d, n, func() float64 { return math.Float64frombits(d.u64()) })
	default:
		return readN(d, n, func() any { return d.array(depth + 1) })
	}
}

func readN[T any](d *decoder, n uint64, next func() T) []T {
	s := make([]T, 0, min(n, maxPrealloc))
	for i := uint64(0); i < n && d.err == nil; i++ {
		s = append(s, next())
	}
	return s
}
