package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/hostfog/gguf/internal/half"
)

var dequantizers = map[Type]func(dst []float32, block []byte){
	TypeF32: func(dst []float32, b []byte) {
		dst[0] = math.Float32frombits(binary.LittleEndian.Uint32(b))
	},
	TypeF16: func(dst []float32, b []byte) {
		dst[0] = half.F16(binary.LittleEndian.Uint16(b))
	},
	TypeBF16: func(dst []float32, b []byte) {
		dst[0] = half.BF16(binary.LittleEndian.Uint16(b))
	},
	TypeQ8_0: func(dst []float32, b []byte) {
		d := half.F16(binary.LittleEndian.Uint16(b))
		for i, q := range b[2:] {
			dst[i] = float32(int8(q)) * d
		}
	},
	TypeQ4_0: func(dst []float32, b []byte) {
		d := half.F16(binary.LittleEndian.Uint16(b))
		for i, q := range b[2:] {
			dst[i] = float32(int(q&0xf)-8) * d
			dst[i+16] = float32(int(q>>4)-8) * d
		}
	},
}

// Dequantize decodes data of type t to float32. F32, F16, BF16, Q8_0 and
// Q4_0 are supported; other types return ErrUnsupported.
func Dequantize(t Type, data []byte) ([]float32, error) {
	fn, ok := dequantizers[t]
	if !ok {
		return nil, fmt.Errorf("%w: %v", ErrUnsupported, t)
	}
	info := typeInfo[t]
	if uint64(len(data))%info.bytes != 0 {
		return nil, fmt.Errorf("gguf: %d bytes is not a whole number of %v blocks", len(data), t)
	}
	blocks := uint64(len(data)) / info.bytes
	out := make([]float32, blocks*info.block)
	for i := uint64(0); i < blocks; i++ {
		fn(out[i*info.block:], data[i*info.bytes:(i+1)*info.bytes])
	}
	return out, nil
}
