package gguf

import "math"

// KV is a metadata entry. Value holds one of uint8, int8, uint16, int16,
// uint32, int32, uint64, int64, float32, float64, bool, string, a slice of
// one of those types, or []any for an array of arrays.
type KV struct {
	Key   string
	Value any
}

// Value returns the raw value stored under key.
func (f *File) Value(key string) (any, bool) {
	i, ok := f.kv[key]
	if !ok {
		return nil, false
	}
	return f.Metadata[i].Value, true
}

// String returns the string stored under key.
func (f *File) String(key string) (string, bool) {
	v, _ := f.Value(key)
	s, ok := v.(string)
	return s, ok
}

// Strings returns the string array stored under key.
func (f *File) Strings(key string) ([]string, bool) {
	v, _ := f.Value(key)
	s, ok := v.([]string)
	return s, ok
}

// Bool returns the bool stored under key.
func (f *File) Bool(key string) (bool, bool) {
	v, _ := f.Value(key)
	b, ok := v.(bool)
	return b, ok
}

// Uint returns the integer stored under key, of any width or signedness, if
// it is non-negative.
func (f *File) Uint(key string) (uint64, bool) {
	v, _ := f.Value(key)
	switch v := v.(type) {
	case uint8:
		return uint64(v), true
	case uint16:
		return uint64(v), true
	case uint32:
		return uint64(v), true
	case uint64:
		return v, true
	}
	n, ok := f.Int(key)
	if !ok || n < 0 {
		return 0, false
	}
	return uint64(n), true
}

// Int returns the integer stored under key, of any width or signedness, if
// it fits in an int64.
func (f *File) Int(key string) (int64, bool) {
	v, _ := f.Value(key)
	switch v := v.(type) {
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	case uint8:
		return int64(v), true
	case uint16:
		return int64(v), true
	case uint32:
		return int64(v), true
	case uint64:
		if v <= math.MaxInt64 {
			return int64(v), true
		}
	}
	return 0, false
}

// Float returns the float32 or float64 stored under key.
func (f *File) Float(key string) (float64, bool) {
	v, _ := f.Value(key)
	switch v := v.(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	}
	return 0, false
}

// Architecture returns general.architecture, such as "llama".
func (f *File) Architecture() string {
	s, _ := f.String("general.architecture")
	return s
}

// Name returns general.name.
func (f *File) Name() string {
	s, _ := f.String("general.name")
	return s
}

// ContextLength returns the training context length, <arch>.context_length.
func (f *File) ContextLength() uint64 {
	n, _ := f.Uint(f.Architecture() + ".context_length")
	return n
}

// FileType returns general.file_type.
func (f *File) FileType() (FileType, bool) {
	n, ok := f.Uint("general.file_type")
	if !ok || n > math.MaxUint32 {
		return 0, false
	}
	return FileType(n), true
}
