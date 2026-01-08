package dksplit

import (
"fmt"
"strings"
"testing"
)

func TestAccuracy(t *testing.T) {
splitter, err := New("models")
if err != nil {
t.Fatalf("Failed to create splitter: %v", err)
}
defer splitter.Close()

tests := []struct {
input    string
expected []string
}{
// Brand names
{"chatgptlogin", []string{"chatgpt", "login"}},
{"microsoftoffice", []string{"microsoft", "office"}},
{"openaiapi", []string{"openai", "api"}},

// Tech terms
{"kubernetescluster", []string{"kubernetes", "cluster"}},
{"dockercontainer", []string{"docker", "container"}},
{"machinelearning", []string{"machine", "learning"}},

// Multi-language (Pinyin)
{"beijingdaxue", []string{"beijing", "daxue"}},
{"pinduoduo", []string{"pin", "duo", "duo"}},

// Euro languages
{"mercibeaucoup", []string{"merci", "beaucoup"}},
{"gutenmorgen", []string{"guten", "morgen"}},
{"buenosdias", []string{"buenos", "dias"}},

// Mixed
{"helloworld", []string{"hello", "world"}},
{"goodmorning", []string{"good", "morning"}},
}

passed := 0
failed := 0

fmt.Printf("\n=== Accuracy Test ===\n")
fmt.Printf("%-20s | %-25s | %-25s | %s\n", "Input", "Expected", "Got", "Status")
fmt.Println(strings.Repeat("-", 85))

for _, tc := range tests {
result, err := splitter.Split(tc.input)
if err != nil {
t.Errorf("Split(%q) error: %v", tc.input, err)
failed++
continue
}

status := "✓ PASS"
if !equal(result, tc.expected) {
status = "✗ FAIL"
failed++
} else {
passed++
}

fmt.Printf("%-20s | %-25s | %-25s | %s\n",
tc.input,
fmt.Sprintf("%v", tc.expected),
fmt.Sprintf("%v", result),
status)
}

fmt.Println(strings.Repeat("-", 85))
fmt.Printf("Total: %d | Passed: %d | Failed: %d | Accuracy: %.1f%%\n",
passed+failed, passed, failed, float64(passed)/float64(passed+failed)*100)
}
