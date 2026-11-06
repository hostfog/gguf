package gguf

import "fmt"

// Type is a ggml tensor type.
type Type uint32

const (
	TypeF32     Type = 0
	TypeF16     Type = 1
	TypeQ4_0    Type = 2
	TypeQ4_1    Type = 3
	TypeQ5_0    Type = 6
	TypeQ5_1    Type = 7
	TypeQ8_0    Type = 8
	TypeQ8_1    Type = 9
	TypeQ2_K    Type = 10
	TypeQ3_K    Type = 11
	TypeQ4_K    Type = 12
	TypeQ5_K    Type = 13
	TypeQ6_K    Type = 14
	TypeQ8_K    Type = 15
	TypeIQ2_XXS Type = 16
	TypeIQ2_XS  Type = 17
	TypeIQ3_XXS Type = 18
	TypeIQ1_S   Type = 19
	TypeIQ4_NL  Type = 20
	TypeIQ3_S   Type = 21
	TypeIQ2_S   Type = 22
	TypeIQ4_XS  Type = 23
	TypeI8      Type = 24
	TypeI16     Type = 25
	TypeI32     Type = 26
	TypeI64     Type = 27
	TypeF64     Type = 28
	TypeIQ1_M   Type = 29
	TypeBF16    Type = 30
	TypeTQ1_0   Type = 34
	TypeTQ2_0   Type = 35
	TypeMXFP4   Type = 39
)

const qkK = 256

var typeInfo = map[Type]struct {
	name  string
	block uint64 // elements per block
	bytes uint64 // bytes per block
}{
	TypeF32:     {"F32", 1, 4},
	TypeF16:     {"F16", 1, 2},
	TypeQ4_0:    {"Q4_0", 32, 2 + 16},
	TypeQ4_1:    {"Q4_1", 32, 2 + 2 + 16},
	TypeQ5_0:    {"Q5_0", 32, 2 + 4 + 16},
	TypeQ5_1:    {"Q5_1", 32, 2 + 2 + 4 + 16},
	TypeQ8_0:    {"Q8_0", 32, 2 + 32},
	TypeQ8_1:    {"Q8_1", 32, 4 + 4 + 32},
	TypeQ2_K:    {"Q2_K", qkK, 2 + 2 + qkK/16 + qkK/4},
	TypeQ3_K:    {"Q3_K", qkK, 2 + qkK/4 + qkK/8 + 12},
	TypeQ4_K:    {"Q4_K", qkK, 2 + 2 + qkK/2 + 12},
	TypeQ5_K:    {"Q5_K", qkK, 2 + 2 + qkK/2 + qkK/8 + 12},
	TypeQ6_K:    {"Q6_K", qkK, 2 + qkK/2 + qkK/4 + qkK/16},
	TypeQ8_K:    {"Q8_K", qkK, 4 + qkK + qkK/8},
	TypeIQ2_XXS: {"IQ2_XXS", qkK, 2 + qkK/4},
	TypeIQ2_XS:  {"IQ2_XS", qkK, 2 + qkK/4 + qkK/32},
	TypeIQ3_XXS: {"IQ3_XXS", qkK, 2 + qkK/4 + qkK/8},
	TypeIQ1_S:   {"IQ1_S", qkK, 2 + qkK/8 + qkK/16},
	TypeIQ4_NL:  {"IQ4_NL", 32, 2 + 16},
	TypeIQ3_S:   {"IQ3_S", qkK, 2 + qkK/4 + qkK/8 + qkK/32 + 4},
	TypeIQ2_S:   {"IQ2_S", qkK, 2 + qkK/4 + qkK/16},
	TypeIQ4_XS:  {"IQ4_XS", qkK, 2 + 2 + qkK/2 + qkK/64},
	TypeI8:      {"I8", 1, 1},
	TypeI16:     {"I16", 1, 2},
	TypeI32:     {"I32", 1, 4},
	TypeI64:     {"I64", 1, 8},
	TypeF64:     {"F64", 1, 8},
	TypeIQ1_M:   {"IQ1_M", qkK, qkK/8 + qkK/16 + qkK/32},
	TypeBF16:    {"BF16", 1, 2},
	TypeTQ1_0:   {"TQ1_0", qkK, 2 + 4*13},
	TypeTQ2_0:   {"TQ2_0", qkK, 2 + qkK/4},
	TypeMXFP4:   {"MXFP4", 32, 1 + 16},
}

func (t Type) String() string {
	if info, ok := typeInfo[t]; ok {
		return info.name
	}
	return fmt.Sprintf("Type(%d)", uint32(t))
}

// BlockSize returns the number of elements in one block of t, or 0 if t is
// unknown.
func (t Type) BlockSize() int { return int(typeInfo[t].block) }

// BlockBytes returns the encoded size in bytes of one block of t, or 0 if t
// is unknown.
func (t Type) BlockBytes() int { return int(typeInfo[t].bytes) }

// FileType is the value of general.file_type: the dominant quantisation of
// the model.
type FileType uint32

var fileTypeNames = map[FileType]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
	15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
	20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
	25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
	30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
	38: "MXFP4_MOE",
}

func (t FileType) String() string {
	if name, ok := fileTypeNames[t]; ok {
		return name
	}
	return fmt.Sprintf("FileType(%d)", uint32(t))
}
