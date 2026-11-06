package gguf

import (
	"bytes"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestDequantize(t *testing.T) {
	q8 := []byte{0x00, 0x38} // d = 0.5
	for i := range 32 {
		q8 = append(q8, byte(int8(i-16)))
	}
	q8want := make([]float32, 32)
	for i := range q8want {
		q8want[i] = float32(i-16) * 0.5
	}

	q4 := []byte{0x00, 0x40} // d = 2
	q4want := make([]float32, 32)
	for i := range 16 {
		lo, hi := i, 15-i
		q4 = append(q4, byte(lo|hi<<4))
		q4want[i] = float32(lo-8) * 2
		q4want[i+16] = float32(hi-8) * 2
	}

	tests := []struct {
		name string
		typ  Type
		data []byte
		want []float32
	}{
		{"f32", TypeF32, []byte{0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0}, []float32{1, -2}},
		{"f16", TypeF16, []byte{0x00, 0x3c, 0x00, 0xc0, 0xff, 0x7b, 0x01, 0x00, 0x00, 0x80}, []float32{1, -2, 65504, 0x1p-24, float32(math.Copysign(0, -1))}},
		{"f16 inf", TypeF16, []byte{0x00, 0x7c, 0x00, 0xfc}, []float32{float32(math.Inf(1)), float32(math.Inf(-1))}},
		{"bf16", TypeBF16, []byte{0x80, 0x3f, 0x00, 0xc0}, []float32{1, -2}},
		{"q8_0", TypeQ8_0, q8, q8want},
		{"q4_0", TypeQ4_0, q4, q4want},
		{"empty", TypeQ4_0, nil, []float32{}},
	}
	for _, tt := range tests {
		got, err := Dequantize(tt.typ, tt.data)
		if err != nil {
			t.Errorf("%s: %v", tt.name, err)
			continue
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestDequantizeNaN(t *testing.T) {
	got, _ := Dequantize(TypeF16, []byte{0x01, 0x7e})
	if !math.IsNaN(float64(got[0])) {
		t.Errorf("got %v, want NaN", got[0])
	}
}

func TestDequantizeErrors(t *testing.T) {
	if _, err := Dequantize(TypeQ4_K, make([]byte, 144)); !errors.Is(err, ErrUnsupported) {
		t.Errorf("Q4_K: err = %v, want ErrUnsupported", err)
	}
	if _, err := Dequantize(TypeQ8_0, make([]byte, 33)); err == nil {
		t.Error("expected error for partial block")
	}

	f, err := parse(build(t, nil, []Tensor{{Name: "k", Shape: []uint64{256}, Type: TypeQ4_K, Data: bytes.Repeat([]byte{0}, 144)}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Float32s("k"); !errors.Is(err, ErrUnsupported) {
		t.Errorf("Float32s(Q4_K): err = %v, want ErrUnsupported", err)
	}
}
