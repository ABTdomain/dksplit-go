# DKSplit-Go

Go implementation of [DKSplit](https://github.com/ABTdomain/dksplit) - fast word segmentation for text without spaces.

Built with BiLSTM-CRF model and ONNX Runtime.

## Performance

| CPU | Mode | QPS |
|-----|------|-----|
| Intel Core i9-14900K | Single | ~1,700/s |
| Intel Core i9-14900K | Batch | ~7,000/s |
| Intel Core i9-9900K | Single | ~1,000/s |
| Intel Core i9-9900K | Batch | ~3,000/s |

Batch mode is **4.6x** faster than single mode.

Compared to Python version:
- Single: **2.7x** faster
- Batch: **5.6x** faster

## Install
```bash
go get github.com/ABTdomain/dksplit-go
```

## Usage
```go
package main

import (
    "fmt"
    "log"

    dksplit "github.com/ABTdomain/dksplit-go"
)

func main() {
    splitter, err := dksplit.New("models")
    if err != nil {
        log.Fatal(err)
    }
    defer splitter.Close()

    // Single
    result, _ := splitter.Split("chatgptlogin")
    fmt.Println(result)
    // Output: [chatgpt login]

    // Batch
    results, _ := splitter.SplitBatch([]string{"openaikey", "microsoftoffice"}, 256)
    fmt.Println(results)
    // Output: [[openai key] [microsoft office]]
}
```

## Examples

| Input | Output |
|-------|--------|
| chatgptlogin | chatgpt login |
| kubernetescluster | kubernetes cluster |
| microsoftoffice | microsoft office |
| mercibeaucoup | merci beaucoup |
| gutenmorgen | guten morgen |

## Real World Benchmark

Tested on [Majestic Million](https://majestic.com/reports/majestic-million) domains:

| Input | Output |
|-------|--------|
| amitriptylineinfo | amitriptyline info |
| autoriteprotectiondonnees | autorite protection donnees |
| mountaingoatsoftware | mountain goat software |
| psychologytoday | psychology today |
| affordablecollegesonline | affordable colleges online |
| stephenwolfram | stephen wolfram |
| ralphlauren | ralphlauren |
| m12ivermectin | m12i vermectin |

Run benchmark yourself:
```bash
wget https://downloads.majestic.com/majestic_million.csv -O top-1m.csv
go test -v -run TestRealWorldBenchmark
```

Results on Intel Core i9-9900K:
- Dataset: 10,000 unique domains (length > 10, no hyphens)
- QPS: 3,175/s

## Requirements

- Go 1.21+
- Linux x64

## Links

- Website: [ABTdomain.com](https://ABTdomain.com)
- Use Case: [domainkits.com](https://domainkits.com)
- Python version: [github.com/ABTdomain/dksplit](https://github.com/ABTdomain/dksplit)
- Documentation: [dksplit.readthedocs.io](https://dksplit.readthedocs.io)
- PyPI: [pypi.org/project/dksplit](https://pypi.org/project/dksplit)

## License

MIT