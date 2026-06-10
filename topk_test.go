package dksplit

import (
	"strings"
	"testing"
)

// TestSplitTopKRank1MatchesSplit checks the core invariant: the rank-1
// candidate from SplitTopK must always equal the output of Split.
func TestSplitTopKRank1MatchesSplit(t *testing.T) {
	splitter, err := New("models")
	if err != nil {
		t.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	inputs := []string{
		"chatgptlogin",
		"openaikey",
		"microsoftoffice",
		"kubernetescluster",
		"machinelearningengineer",
		"iphone15promax",
		"expertsexchange",
		"helloworld",
		"mercibeaucoup",
		"noranite",
		"pikahug",
		"tiantian5",
	}

	for _, input := range inputs {
		single, err := splitter.Split(input)
		if err != nil {
			t.Fatalf("Split(%q) error: %v", input, err)
		}

		topk, err := splitter.SplitTopK(input, 5)
		if err != nil {
			t.Fatalf("SplitTopK(%q, 5) error: %v", input, err)
		}
		if len(topk) == 0 {
			t.Fatalf("SplitTopK(%q, 5) returned no candidates", input)
		}
		if !equal(topk[0], single) {
			t.Errorf("SplitTopK(%q) rank-1 = %v, want Split result %v", input, topk[0], single)
		}
	}
}

// TestSplitTopKCandidates checks structural properties of the candidate
// set: at most k candidates, all unique, and each candidate reassembles
// to the (lowercased) input.
func TestSplitTopKCandidates(t *testing.T) {
	splitter, err := New("models")
	if err != nil {
		t.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	inputs := []string{"chatgptlogin", "noranite", "pikahug", "schwarzwald", "openaikey"}

	for _, input := range inputs {
		for _, k := range []int{1, 2, 3, 5, 8} {
			cands, err := splitter.SplitTopK(input, k)
			if err != nil {
				t.Fatalf("SplitTopK(%q, %d) error: %v", input, k, err)
			}
			if len(cands) == 0 || len(cands) > k {
				t.Errorf("SplitTopK(%q, %d) returned %d candidates", input, k, len(cands))
			}

			seen := make(map[string]bool)
			for _, c := range cands {
				joined := strings.Join(c, " ")
				if seen[joined] {
					t.Errorf("SplitTopK(%q, %d) duplicate candidate %v", input, k, c)
				}
				seen[joined] = true

				if strings.Join(c, "") != strings.ToLower(input) {
					t.Errorf("SplitTopK(%q, %d) candidate %v does not reassemble to input", input, k, c)
				}
			}
		}
	}
}

// TestSplitTopKAmbiguous checks that genuinely ambiguous inputs yield
// multiple candidates, including both the kept-whole and the split reading.
func TestSplitTopKAmbiguous(t *testing.T) {
	splitter, err := New("models")
	if err != nil {
		t.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	for _, input := range []string{"noranite", "pikahug"} {
		cands, err := splitter.SplitTopK(input, 5)
		if err != nil {
			t.Fatalf("SplitTopK(%q, 5) error: %v", input, err)
		}
		if len(cands) < 2 {
			t.Fatalf("SplitTopK(%q, 5) returned %d candidates, want >= 2", input, len(cands))
		}

		hasWhole := false
		hasSplit := false
		for _, c := range cands {
			if len(c) == 1 {
				hasWhole = true
			} else {
				hasSplit = true
			}
		}
		if !hasWhole || !hasSplit {
			t.Errorf("SplitTopK(%q, 5) = %v, want both a kept-whole and a split candidate", input, cands)
		}
	}
}

// TestSplitTopKEdgeCases mirrors the Python test.py edge cases: empty
// input, single character, two characters, and invalid k.
func TestSplitTopKEdgeCases(t *testing.T) {
	splitter, err := New("models")
	if err != nil {
		t.Fatalf("Failed to create splitter: %v", err)
	}
	defer splitter.Close()

	empty, err := splitter.SplitTopK("", 3)
	if err != nil {
		t.Fatalf("SplitTopK(\"\", 3) error: %v", err)
	}
	if len(empty) != 0 {
		t.Errorf("SplitTopK(\"\", 3) = %v, want empty", empty)
	}

	one, err := splitter.SplitTopK("a", 3)
	if err != nil {
		t.Fatalf("SplitTopK(\"a\", 3) error: %v", err)
	}
	if len(one) != 1 || !equal(one[0], []string{"a"}) {
		t.Errorf("SplitTopK(\"a\", 3) = %v, want [[a]]", one)
	}

	two, err := splitter.SplitTopK("ab", 3)
	if err != nil {
		t.Fatalf("SplitTopK(\"ab\", 3) error: %v", err)
	}
	if len(two) != 2 {
		t.Errorf("SplitTopK(\"ab\", 3) = %v, want exactly 2 candidates", two)
	}

	if _, err := splitter.SplitTopK("abc", 0); err == nil {
		t.Error("SplitTopK(\"abc\", 0) should return an error")
	}

	// Split3 / Split5 are thin wrappers; spot-check they bound k.
	three, err := splitter.Split3("chatgptlogin")
	if err != nil {
		t.Fatalf("Split3 error: %v", err)
	}
	if len(three) == 0 || len(three) > 3 {
		t.Errorf("Split3 returned %d candidates", len(three))
	}
	five, err := splitter.Split5("chatgptlogin")
	if err != nil {
		t.Fatalf("Split5 error: %v", err)
	}
	if len(five) == 0 || len(five) > 5 {
		t.Errorf("Split5 returned %d candidates", len(five))
	}
}
