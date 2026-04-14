# DKSplit-go

> **v0.3.1** — Model upgraded to EuroHPC infrastructure (Leonardo Booster, NVIDIA A100). ~3% accuracy improvement over v0.2.x on real-world domains.

Go implementation of [DKSplit](https://github.com/ABTdomain/dksplit) - fast word segmentation for text without spaces.

Built with BiLSTM-CRF model (9.47M parameters) and ONNX Runtime. The Go and Python versions use the same model and produce identical results.

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

## Benchmark

### Dataset

1,000 domains randomly sampled from the [Newly Registered Domains Database (NRDS)](https://domainkits.com/download/nrds) (.com feed, April 8, 2026). No filtering or cherry-picking. Ground truth was established through multi-model cross-validation (BiLSTM, Qwen 9B LoRA, Gemma 31B) and human audit.

The dataset and evaluation script are available on [GitHub](https://github.com/ABTdomain/dksplit/tree/main/benchmark).

### Results

Accuracy on 1,000 randomly sampled real-world .com domains, human-audited ground truth:

| Model | Accuracy |
|---|---|
| **DKSplit v0.3.1** | **85.0%** |
| DKSplit v0.2.x | 82.8% |
| WordSegment | 54.0% |
| WordNinja | 46.1% |

> **Note:** The accuracy above is measured against a single reference segmentation. Domain names are inherently ambiguous. For example, `tiantian5` could be `tiantian 5` (Chinese compound name) or `tian tian 5` (two separate syllables); `noranite` could be `nora nite` or an intact brand; `pikahug` could be `pika hug` or an intact brand name. Our audit found ~5% of test samples have multiple valid segmentations. Accounting for these, effective accuracy is closer to **90%**.

### Comparison

| Input | DKSplit v0.3.1 | WordSegment | WordNinja |
|---|---|---|---|
| `chatgptprompts` | **chatgpt prompts** | chat gpt prompts | chat gp t prompts |
| `tensorflowserving` | **tensorflow serving** | tensor flow serving | tensor flow serving |
| `spotifywrapped` | **spotify wrapped** | spot if y wrapped | spot if y wrapped |
| `ethereumwallet` | **ethereum wallet** | e there um wallet | e there um wallet |
| `cloudflarecdn` | **cloudflare cdn** | cloud flare cdn | cloud flare cd n |
| `kubernetescluster` | **kubernetes cluster** | ku bernet es cluster | ku berne tes cluster |
| `hackathonwinners` | **hackathon winners** | hackathon winners | hack a th on winners |
| `whatsappstatus` | **whatsapp status** | what sapp status | what s app status |
| `drwatsonai` | **dr watson ai** | dr watson a i | dr watson a i |
| `escribirenvozalta` | **escribir en voz alta** | escribir env oz alta | es crib ire nv oz alta |
| `tuvasou` | **tu vas ou** | tuva sou | tuva so u |
| `candidiasenuncamais` | **candidiase nunca mais** | candid iase nunca mais | can didi as e nun cama is |
| `robertdeniro` | **robert de niro** | robert deniro | robert deniro |
| `mercibeaucoup` | **merci beaucoup** | merci beaucoup | mer ci beau coup |

## Features

- **Brand-aware:** Recognizes thousands of brands, tech products, and proper nouns
- **Multilingual:** Handles English, French, German, Spanish, and romanized text
- **Lightweight:** 9 MB model, ONNX Runtime inference
- **Offline:** No API keys, no internet required

## Limitations

- **Characters:** Only `a-z` and `0-9`. Input is automatically lowercased.
- **Max length:** 64 characters.
- **Script:** Latin script only.
- **Platform:** Linux x64 (ONNX Runtime shared library included).

## Requirements

- Go 1.21+
- Linux x64

## Links

- Website: [domainkits.com](https://domainkits.com), [ABTdomain.com](https://ABTdomain.com)
- Python version: [github.com/ABTdomain/dksplit](https://github.com/ABTdomain/dksplit)
- PyPI: [pypi.org/project/dksplit](https://pypi.org/project/dksplit)
- Hugging Face: [huggingface.co/ABTdomain/dksplit](https://huggingface.co/ABTdomain/dksplit)

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

**Please attribute as:** DKsplit by [ABTdomain](https://abtdomain.com)

## Acknowledgements

<a href="https://eurohpc-ju.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eurohpc-logo.png" alt="EuroHPC JU" width="80"></a> &nbsp; <a href="https://commission.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eu-cofunded-logo.png" alt="Co-funded by the EU" width="200"></a>

The v0.3.1 model was trained on the [Leonardo Booster](https://www.hpc.cineca.it/systems/hardware/leonardo/) supercomputer at CINECA, Italy, with computing resources provided by the [EuroHPC Joint Undertaking](https://eurohpc-ju.europa.eu/) through the Playground Access program (project AIFAC_P02_281). We thank EuroHPC JU for enabling SMEs to explore new possibilities with world-class HPC infrastructure.
