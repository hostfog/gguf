package gguf_test

import (
	"bytes"
	"fmt"

	"github.com/hostfog/gguf"
)

func Example() {
	var buf bytes.Buffer
	err := gguf.Write(&buf, []gguf.KV{
		{Key: "general.architecture", Value: "llama"},
		{Key: "llama.context_length", Value: uint32(4096)},
		{Key: "tokenizer.ggml.tokens", Value: []string{"<s>", "</s>"}},
	}, []gguf.Tensor{
		{Name: "output.weight", Shape: []uint64{2, 2}, Type: gguf.TypeF16, Data: []byte{0, 0x3c, 0, 0x40, 0, 0x42, 0, 0x44}},
	})
	if err != nil {
		panic(err)
	}

	f, err := gguf.Read(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		panic(err)
	}
	tokens, _ := f.Strings("tokenizer.ggml.tokens")
	weights, _ := f.Float32s("output.weight")

	fmt.Println(f.Architecture(), f.ContextLength(), f.ParameterCount())
	fmt.Println(tokens)
	fmt.Println(weights)
	// Output:
	// llama 4096 4
	// [<s> </s>]
	// [1 2 3 4]
}

func ExampleFile_TensorReader() {
	var buf bytes.Buffer
	gguf.Write(&buf, nil, []gguf.Tensor{
		{Name: "bias", Shape: []uint64{4}, Type: gguf.TypeI8, Data: []byte{1, 2, 3, 4}},
	})
	f, _ := gguf.Read(bytes.NewReader(buf.Bytes()), int64(buf.Len()))

	t, _ := f.Lookup("bias")
	r, _ := f.TensorReader("bias")
	data := make([]byte, r.Size())
	r.ReadAt(data, 0)

	fmt.Println(t.Type, t.Shape, data)
	// Output: I8 [4] [1 2 3 4]
}
