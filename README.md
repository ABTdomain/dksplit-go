# DKSplit-go

> **v1.0.0**: First stable release. The model is frozen and the API is stable. Includes the top-k API: `SplitTopK` / `Split3` / `Split5` return the k best candidate segmentations, ranked.

Go implementation of [DKSplit](https://github.com/ABTdomain/dksplit), fast word segmentation for text without spaces.

Built with a BiLSTM-CRF model (9.47M parameters) and ONNX Runtime. The Go and Python versions load the same model file and implement the same decoder, and produce identical results.

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

    // Top-k candidates, best first
    candidates, _ := splitter.Split3("noranite")
    fmt.Println(candidates)
    // Output: [[nora nite] [noranite] [nor anite]]

    candidates, _ = splitter.SplitTopK("chatgptlogin", 3) // any k
    fmt.Println(candidates)
    // Output: [[chatgpt login] [chatgptlogin] [chatgpt log in]]
}
```

## What's New in v1.0.0

First stable release. The model is frozen and the public API (`Split`, `SplitBatch`, `SplitTopK` / `Split3` / `Split5`) is stable.

### From one answer to a set of answers

A single segmentation has to commit to one reading, but real domains are often genuinely ambiguous (is `noranite` a brand, or `nora nite`?). Instead of forcing one fixed answer, top-k returns a set of ranked candidates. For some use cases that is simply the better solution: when the input is ambiguous, a small ranked set is more honest, and more useful, than a single guess that has to be right or wrong.

### Why the model is frozen

From an engineering standpoint it is already the best trade-off between accuracy and speed: a 9 MB CPU-only model with no GPU or external dependencies. Larger models score higher but cost hundreds of times the compute (see the [EuroHPC blog post](https://abtdomain.com/blog/2026/06/dksplit-on-eurohpc-unlocking-a-4b-models-knowledge-through-chain-of-thought/)), so we froze it as a stable baseline and put further gains into the candidate layer rather than a heavier model.

### The top-k API

`SplitTopK(text, k)` returns the k best candidate segmentations instead of just one. `Split3` and `Split5` are shorthands for k=3 and k=5. `Split` and `SplitBatch` are unchanged.

Candidates are decoded with k-best Viterbi over the same CRF: no model change, no new dependencies, and only a small speed overhead. Inputs with fewer than k possible segmentations return fewer candidates. The rank-1 candidate always equals the output of `Split`.

Across both benchmarks, an acceptable segmentation (`truth` or `might_right`) is present within the top-k candidates far more often than in the single best output:

| Benchmark | top-1 | top-3 | top-5 |
|---|---|---|---|
| 1,000 samples | 91.5% | 98.5% | 99.3% |
| 5,000 samples | 90.4% | 97.8% | 99.0% |

## Performance

| CPU | Mode | QPS |
|-----|------|-----|
| Intel Core i9-14900K | Single | ~1,700/s |
| Intel Core i9-14900K | Batch | ~7,000/s |
| Intel Core i9-9900K | Single | ~1,000/s |
| Intel Core i9-9900K | Batch | ~3,000/s |

Batch mode is **4.6x** faster than single mode.

Compared to the Python version:
- Single: **2.7x** faster
- Batch: **5.6x** faster

## Benchmark

### Dataset

1,000 hand-audited domain prefixes drawn from the [Newly Registered Domains Database (NRDS)](https://domainkits.com/download/nrds) (.com feed). No filtering or cherry-picking on segmentation difficulty. Ground truth was established through multi-model cross-validation (BiLSTM, Qwen 9B LoRA, Gemma 31B) and human audit. Each row provides a primary `truth` and an optional `might_right` field for genuinely ambiguous cases (e.g. brand-versus-compound).

The dataset and evaluation script are available in the [Python repository](https://github.com/ABTdomain/dksplit/tree/main/benchmark). The numbers below are measured with that harness; the Go build loads the same model weights and implements the same decoder, and produces the same segmentations.

This benchmark is multi-audited, but it is only a reference point. Human language is endlessly varied. No fixed test set of any size can cover every brand coinage, multilingual compound, or naming convention that shows up in real registrations, and we make no claim of 100% coverage. The honest way to judge DKSplit is on your own data: download a fresh batch of newly registered domains from [domainkits.com/download/nrds](https://domainkits.com/download/nrds) (free, domain-name-only files) and run them through it.

### Results

| Model | Strict EM | Lenient EM |
|---|---|---|
| **DKSplit v1.0.0** | **86.5%** | **91.5%** |
| WordSegment | 65.2% | 69.5% |
| WordNinja | 51.0% | 54.0% |

Strict EM counts only exact matches against `truth`. Lenient EM also accepts the `might_right` alternative when present. DKSplit outperforms WordSegment by 21+ percentage points and WordNinja by 35+ percentage points on both measures.

> **Note:** Domain names are inherently ambiguous. For example, `tiantian5` could be `tiantian 5` (Chinese compound name) or `tian tian 5` (two separate syllables); `noranite` could be `nora nite` or an intact brand; `pikahug` could be `pika hug` or an intact brand name. The Lenient EM column above reflects the cases where multiple segmentations are accepted as correct.

### Comparison

| Input | DKSplit v1.0.0 | WordSegment | WordNinja |
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
- **Top-k candidates:** `SplitTopK` / `Split3` / `Split5` return ranked alternative segmentations

## Limitations

- **Characters:** Only `a-z` and `0-9`. Input is automatically lowercased.
- **Max length:** 64 characters.
- **Script:** Latin script only.
- **Ambiguity:** Some inputs are genuinely ambiguous. `Split` optimizes for the most common interpretation; use the top-k API when your pipeline can handle multiple candidates.
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

**Attribution required.** Any public or production use of DKSplit must visibly credit **DKSplit from [ABTdomain.com](https://abtdomain.com)**, for example in your README, documentation, about page, or API response metadata. This requirement is in addition to the Apache-2.0 license terms.

## Acknowledgements

<a href="https://eurohpc-ju.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eurohpc-logo.png" alt="EuroHPC JU" width="80"></a> &nbsp; <a href="https://commission.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eu-cofunded-logo.png" alt="Co-funded by the EU" width="200"></a>

The model was trained on the [Leonardo Booster](https://www.hpc.cineca.it/systems/hardware/leonardo/) supercomputer at CINECA, Italy, with computing resources provided by the [EuroHPC Joint Undertaking](https://eurohpc-ju.europa.eu/) through the Playground Access program (project AIFAC_P02_281). We thank EuroHPC JU for enabling SMEs to explore new possibilities with world-class HPC infrastructure.
