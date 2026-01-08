package dksplit

import (
"fmt"
"testing"
"time"
)

func TestBenchmarkSingle(t *testing.T) {
splitter, err := New("models")
if err != nil {
t.Fatalf("Failed to create splitter: %v", err)
}
defer splitter.Close()

testCases := []string{
"chatgptlogin",
"microsoftoffice",
"kubernetescluster",
"helloworld",
"openaiapi",
}

// Warmup
for i := 0; i < 100; i++ {
splitter.Split(testCases[i%len(testCases)])
}

// Benchmark single
iterations := 10000
start := time.Now()
for i := 0; i < iterations; i++ {
splitter.Split(testCases[i%len(testCases)])
}
elapsed := time.Since(start)

qps := float64(iterations) / elapsed.Seconds()
avgLatency := elapsed.Microseconds() / int64(iterations)

fmt.Printf("\n=== Single Mode Benchmark ===\n")
fmt.Printf("Iterations: %d\n", iterations)
fmt.Printf("Total time: %v\n", elapsed)
fmt.Printf("QPS: %.2f/s\n", qps)
fmt.Printf("Avg latency: %d μs\n", avgLatency)
}

func TestBenchmarkBatch(t *testing.T) {
splitter, err := New("models")
if err != nil {
t.Fatalf("Failed to create splitter: %v", err)
}
defer splitter.Close()

// Generate test data
baseTexts := []string{
"chatgptlogin",
"microsoftoffice",
"kubernetescluster",
"helloworld",
"openaiapi",
"machinelearning",
"deeplearning",
"naturallanguage",
"computervision",
"neuralnetwork",
}

// Create batch of 1000 items
batchSize := 1000
texts := make([]string, batchSize)
for i := 0; i < batchSize; i++ {
texts[i] = baseTexts[i%len(baseTexts)]
}

// Warmup
splitter.SplitBatch(texts, 256)

// Benchmark batch
iterations := 100
totalItems := iterations * batchSize

start := time.Now()
for i := 0; i < iterations; i++ {
splitter.SplitBatch(texts, 256)
}
elapsed := time.Since(start)

qps := float64(totalItems) / elapsed.Seconds()
avgLatency := elapsed.Microseconds() / int64(iterations)

fmt.Printf("\n=== Batch Mode Benchmark ===\n")
fmt.Printf("Batch size: %d\n", batchSize)
fmt.Printf("Iterations: %d\n", iterations)
fmt.Printf("Total items: %d\n", totalItems)
fmt.Printf("Total time: %v\n", elapsed)
fmt.Printf("QPS: %.2f/s\n", qps)
fmt.Printf("Avg batch latency: %d μs\n", avgLatency)
}

func TestBenchmarkComparison(t *testing.T) {
splitter, err := New("models")
if err != nil {
t.Fatalf("Failed to create splitter: %v", err)
}
defer splitter.Close()

texts := []string{
"chatgptlogin",
"microsoftoffice",
"kubernetescluster",
"helloworld",
"openaiapi",
}

// Expand to 1000
batch := make([]string, 1000)
for i := range batch {
batch[i] = texts[i%len(texts)]
}

// Single mode for 1000 items
start := time.Now()
for _, text := range batch {
splitter.Split(text)
}
singleTime := time.Since(start)

// Batch mode for 1000 items
start = time.Now()
splitter.SplitBatch(batch, 256)
batchTime := time.Since(start)

fmt.Printf("\n=== Comparison (1000 items) ===\n")
fmt.Printf("Single mode: %v (%.2f/s)\n", singleTime, 1000/singleTime.Seconds())
fmt.Printf("Batch mode:  %v (%.2f/s)\n", batchTime, 1000/batchTime.Seconds())
fmt.Printf("Speedup: %.2fx\n", float64(singleTime)/float64(batchTime))
}
