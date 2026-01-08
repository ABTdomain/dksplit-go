# DKSplit-Go

Go implementation of [DKSplit](https://github.com/ABTdomain/dksplit) - fast word segmentation for text without spaces.

Built with BiLSTM-CRF model and ONNX Runtime.

## Performance

| Mode | QPS | Latency |
|------|-----|---------|
| Single | 2,128/s | 469μs |
| Batch | 9,565/s | - |

Batch mode is **4.6x** faster than single mode.

Compared to Python version:
- Single: **2.7x** faster
- Batch: **5.6x** faster

## Install
```bash
go get github.com/ABTdomain/dksplit-go
```

### ONNX Runtime Dependency

Download and set up ONNX Runtime 1.20.0:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-1.20.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.20.0/lib:$LD_LIBRARY_PATH
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

## Requirements

- Go 1.21+
- ONNX Runtime 1.20.0

## Links

- Website: [domainkits.com](https://domainkits.com), [ABTdomain.com](https://ABTdomain.com)
- Python version: [github.com/ABTdomain/dksplit](https://github.com/ABTdomain/dksplit)
- Documentation: [dksplit.readthedocs.io](https://dksplit.readthedocs.io)
- PyPI: [pypi.org/project/dksplit](https://pypi.org/project/dksplit)

## License

MIT
