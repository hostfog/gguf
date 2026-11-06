# gguf

Read GGUF (llama.cpp) and safetensors model files in pure Go. No cgo, no dependencies.

```sh
go get github.com/hostfog/gguf
go install github.com/hostfog/gguf/cmd/gguf-info@latest
```

## Usage

```go
f, err := gguf.Open("llama-3-8b.Q4_K_M.gguf")
if err != nil {
	log.Fatal(err)
}
defer f.Close()

fmt.Println(f.Architecture(), f.ContextLength(), f.ParameterCount())
vocab, _ := f.Strings("tokenizer.ggml.tokens")

r, _ := f.TensorReader("token_embd.weight")   // raw bytes, read on demand
w, _ := f.Float32s("output_norm.weight")      // F32, F16, BF16, Q8_0, Q4_0
```

safetensors:

```go
f, _ := safetensors.Open("model.safetensors")
defer f.Close()

for _, t := range f.Tensors {
	fmt.Println(t.Name, t.DType, t.Shape)
}
w, _ := f.Float32s("lm_head.weight")          // F32, F16, BF16
```

## Design

- **Header only.** Opening a file parses metadata and tensor info; tensor data is read through `io.SectionReader` when you ask for it.
- **Hostile input is an error, not a panic.** Counts, lengths and offsets are checked against the file size before anything is allocated. The GGUF parser is fuzzed.
- **All ggml types sized.** Byte sizes are known for every current type, including K-quants and IQ types. Dequantization covers F16, BF16, Q8_0 and Q4_0; others return `ErrUnsupported`.
- **GGUF v2 and v3**, little-endian. Big-endian files return `ErrBigEndian`.
- **Writer included.** `gguf.Write` produces valid v3 files, handy for tests and tooling.

## License

MIT
