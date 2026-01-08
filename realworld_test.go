package dksplit

import (
"bufio"
"fmt"
"math/rand"
"os"
"strings"
"testing"
"time"
)

func loadRealData(filepath string, limit int) ([]string, error) {
file, err := os.Open(filepath)
if err != nil {
return nil, err
}
defer file.Close()

var dataset []string
scanner := bufio.NewScanner(file)

// Skip header
scanner.Scan()

for scanner.Scan() {
line := scanner.Text()
parts := strings.Split(line, ",")
if len(parts) >= 3 {
domain := parts[2] // Domain is 3rd column
// Get prefix before first dot
if idx := strings.Index(domain, "."); idx != -1 {
domain = domain[:idx]
}
// Skip if contains hyphen
if strings.Contains(domain, "-") {
continue
}
// Only keep domains longer than 10 chars
if len(domain) > 10 {
dataset = append(dataset, domain)
}
}
if len(dataset) >= limit {
break
}
}
return dataset, scanner.Err()
}

func TestRealWorldBenchmark(t *testing.T) {
splitter, err := New("models")
if err != nil {
t.Fatalf("Failed to create splitter: %v", err)
}
defer splitter.Close()

fmt.Println("Loading dataset...")
dataset, err := loadRealData("top-1m.csv", 10000)
if err != nil || len(dataset) == 0 {
t.Log("No dataset found, using generated data.")
dataset = generateRandomData(10000)
}

fmt.Printf("Dataset size: %d unique items\n", len(dataset))

// Warmup
splitter.SplitBatch(dataset[:100], 256)

// Random sample 100 items
rand.Seed(time.Now().UnixNano())
sampleIdx := rand.Perm(len(dataset))[:100]
sample := make([]string, 100)
for i, idx := range sampleIdx {
sample[i] = dataset[idx]
}

// Preview random 100 results
fmt.Printf("\n=== Sample Results (random 100) ===\n")
fmt.Printf("%-30s | %s\n", "Input", "Output")
fmt.Println(strings.Repeat("-", 70))

preview, _ := splitter.SplitBatch(sample, 256)
for i, result := range preview {
fmt.Printf("%-30s | %s\n", sample[i], strings.Join(result, " "))
}
fmt.Println(strings.Repeat("-", 70))

// Benchmark
batchSize := 256
start := time.Now()

for i := 0; i < len(dataset); i += batchSize {
end := i + batchSize
if end > len(dataset) {
end = len(dataset)
}
splitter.SplitBatch(dataset[i:end], 1000)
}

elapsed := time.Since(start)
qps := float64(len(dataset)) / elapsed.Seconds()

fmt.Printf("\n=== Real World Benchmark ===\n")
fmt.Printf("Total Unique Items: %d\n", len(dataset))
fmt.Printf("Total Time: %v\n", elapsed)
fmt.Printf("Real QPS: %.2f/s\n", qps)
}

func generateRandomData(n int) []string {
var data []string
vocab := []string{"cloud", "flare", "google", "test", "api", "server", "host", "net", "work", "soft"}
for i := 0; i < n; i++ {
s := fmt.Sprintf("%s%s%s%d", vocab[i%len(vocab)], vocab[(i+1)%len(vocab)], vocab[(i+3)%len(vocab)], i)
data = append(data, s)
}
return data
}
