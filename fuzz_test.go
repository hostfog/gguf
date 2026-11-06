package gguf

import (
	"io"
	"testing"
)

func FuzzRead(f *testing.F) {
	f.Add(build(f, nil, nil))
	f.Add(build(f, []KV{
		{"general.architecture", "llama"},
		{"tokens", []string{"a", "b"}},
		{"nested", []any{[]int32{1}}},
	}, []Tensor{
		{Name: "w", Shape: []uint64{32}, Type: TypeQ8_0, Data: make([]byte, 34)},
	}))

	f.Fuzz(func(t *testing.T, data []byte) {
		gf, err := parse(data)
		if err != nil {
			return
		}
		for _, ti := range gf.Tensors {
			sr, err := gf.TensorReader(ti.Name)
			if err != nil {
				t.Fatal(err)
			}
			if n, err := io.Copy(io.Discard, sr); err != nil || uint64(n) != ti.Size() {
				t.Fatalf("tensor %q: read %d of %d bytes: %v", ti.Name, n, ti.Size(), err)
			}
		}
	})
}
