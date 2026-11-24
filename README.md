# EvoIC: Unsupervised Evolutionary Prompt Optimization for Image Clustering

**EvoIC** uses LLM-generated fine-grained prompts and evolutionary search to improve image–text representations for image clustering. It leverages **frozen CLIP**, **WordNet**, and an **unsupervised fitness objective**; prompt selection does not require updating model parameters.

## Framework

![Overall framework of EvoIC](figure2.png)

*Figure 1. Overall framework of EvoIC: fine-grained prompt generation and evolutionary prompt optimization.*

EvoIC maps preliminary image clusters to WordNet nouns and uses an LLM to generate fine-grained prompts. I-TSS samples initial prompt combinations based on image–text similarity. Evolutionary optimization then refines the combinations using an unsupervised fitness score combining semantic consistency, cluster separation, and distribution balance.

## Project Structure

```text
EvoIC/
├── README.md              # Repository overview
├── figure2.png            # Framework figure
├── CLIP.py                # CLIP model wrapper
├── dataLoader.py          # Dataset loading
├── imageEmbedding.py      # Image feature extraction
├── textEmbedding.py       # WordNet noun embeddings
├── selectNouns.py         # Noun mapping
├── promptGenerate.py      # LLM prompt generation
├── PromptOptimization.py  # I-TSS and evolutionary optimization
├── evalMetric.py          # ACC, NMI and ARI
└── data/                  # Datasets and intermediate features
```

## Requirements

- Python 3.10+ and a CUDA-capable GPU
- PyTorch, torchvision, NumPy, SciPy, scikit-learn, pandas, tqdm
- OpenAI CLIP, OpenCLIP, GPU-enabled FAISS, and `openai`
- CLIP ViT-B/32 weights; an OpenAI-compatible LLM API for generating new prompts

Install the required libraries in a compatible CUDA environment. The archive does **not** include a pinned, tested environment file.

## Usage

Run the scripts from the repository root in this order:

```bash
python imageEmbedding.py    # Extract image embeddings
python textEmbedding.py     # Encode WordNet nouns
python selectNouns.py       # Map nouns to preliminary clusters
python promptGenerate.py    # Generate candidate prompts (requires API access)
python PromptOptimization.py  # Encode prompts if embeddings are absent; exits
python PromptOptimization.py  # Run evolutionary optimization
```

Set the dataset, model, and paths inside each script before running. The scripts currently use hard-coded configuration values rather than command-line arguments. For an existing prompt set, you can skip `promptGenerate.py`.

**Datasets:** STL-10, CIFAR-10, CIFAR-20, ImageNet-10, ImageNet-Dogs, and DTD. Raw datasets, complete pretrained features, and the full generated prompt collection are **not included** in the provided archive.

**Default search settings:** population size `N = 20`, evolution iterations `T = 10`, and five generated prompts per preliminary cluster.

## Results

Clustering performance reported in the manuscript with CLIP ViT-B/32 (%, **not independently reproduced from this archive**):

| Dataset | NMI | ACC | ARI |
|---|---:|---:|---:|
| STL-10 | 95.7 | 98.3 | 96.3 |
| CIFAR-10 | 83.5 | 91.9 | 83.3 |
| CIFAR-20 | 62.1 | 61.6 | 44.6 |
| ImageNet-10 | 98.5 | 99.2 | 98.2 |
| ImageNet-Dogs | 86.5 | 89.4 | 81.0 |
| DTD | 63.1 | 51.5 | 34.6 |

## Notes

This is a **research-code snapshot**, not a fully validated reproduction package. In particular, the supplied optimization script currently uses ground-truth labels to align intermediate clustering probabilities when computing a fitness component. That implementation detail must be corrected and the experiments re-evaluated before claiming a strictly label-free reproduction. Do **not** upload API keys or local IDE configuration files.

## Citation

If you find this project useful, please cite:

> Wangwang Li, Na Chen, Qun Chen, Wenjie Liu, and Zhanhuai Li. *Unsupervised Evolutionary Prompt Optimization for Image Clustering*. Manuscript.

A publication-specific BibTeX entry will be added when available.
