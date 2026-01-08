package dksplit

import (
	"testing"
)

func TestSplit(t *testing.T) {
	splitter, err := New("models")
	if err != nil {
		t.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	tests := []struct {
		input    string
		expected []string
	}{
		{"chatgptlogin", []string{"chatgpt", "login"}},
		{"helloworld", []string{"hello", "world"}},
		{"microsoftoffice", []string{"microsoft", "office"}},
	}

	for _, tc := range tests {
		result, err := splitter.Split(tc.input)
		if err != nil {
			t.Errorf("Split(%q) error: %v", tc.input, err)
			continue
		}
		if !equal(result, tc.expected) {
			t.Errorf("Split(%q) = %v, want %v", tc.input, result, tc.expected)
		}
	}
}

func TestSplitBatch(t *testing.T) {
	splitter, err := New("models")
	if err != nil {
		t.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	inputs := []string{
		"chatgptlogin",
		"openaikey",
		"microsoftoffice",
		"helloworld",
	}

	results, err := splitter.SplitBatch(inputs, 256)
	if err != nil {
		t.Fatalf("SplitBatch error: %v", err)
	}

	if len(results) != len(inputs) {
		t.Errorf("SplitBatch returned %d results, want %d", len(results), len(inputs))
	}

	t.Logf("Results: %v", results)
}

func BenchmarkSplit(b *testing.B) {
	splitter, err := New("models")
	if err != nil {
		b.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitter.Split("chatgptlogin")
	}
}

func BenchmarkSplitBatch(b *testing.B) {
	splitter, err := New("models")
	if err != nil {
		b.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	inputs := make([]string, 1000)
	for i := range inputs {
		inputs[i] = "chatgptlogin"
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitter.SplitBatch(inputs, 256)
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}