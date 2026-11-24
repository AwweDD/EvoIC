# EvoIC

**Unsupervised Evolutionary Prompt Optimization for Image Clustering**

This repository contains the research implementation of **EvoIC**, which uses WordNet-derived nouns to guide LLM prompt generation and applies evolutionary search to select fine-grained prompt combinations for image clustering.

## Code and data layout

```text
EvoIC/
├── CLIP.py
├── dataLoader.py
├── imageEmbedding.py
├── textEmbedding.py
├── selectNouns.py
├── promptGenerate.py
├── PromptOptimization.py
├── evalMetric.py
├── README.md
├── figure2.png
```

| Script | Purpose |
| --- | --- |
| `CLIP.py` | CLIP image/text encoder wrapper |
| `dataLoader.py` | Dataset loading and preprocessing |
| `imageEmbedding.py` | Extract and save train/test image embeddings and labels |
| `textEmbedding.py` | Encode WordNet nouns using CLIP |
| `selectNouns.py` | Assign related WordNet nouns to preliminary image clusters |
| `promptGenerate.py` | Generate candidate prompts through an LLM API |
| `PromptOptimization.py` | Initialize and evolve prompt combinations; print clustering metrics |
| `evalMetric.py` | Clustering metric utilities |

## Requirements

A **CUDA-enabled GPU** is required by the supplied scripts. Main Python dependencies include:

- PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, tqdm
- GPU-enabled FAISS (`import faiss`)
- OpenAI CLIP (`import clip`)
- `open_clip_torch` (`import open_clip`)
- `openai` (for the LLM client)

Dependency versions are **not pinned** in the supplied code, and this repository is not a verified one-command installation. Install versions compatible with your Python, CUDA, and PyTorch environment.

## Run the pipeline

Run the following commands from the **repository root**, in order. The scripts do **not** provide a unified command-line interface: edit the `dataset` and other configuration variables inside each script before execution.

| Step | Command | Required input | Output |
| --- | --- | --- | --- |
| 1. WordNet encoding | `python textEmbedding.py` | `data/1_text_embedding/WordNetNouns.csv` | `data/1_text_embedding/WordNetNouns_Embeddings.npy` |
| 2. Image encoding | `python imageEmbedding.py` | Dataset under `data/4_datasets/` or downloaded by torchvision | `data/0_image_embedding/ViT-B-32/{dataset}_image_embedding_{train,test}.npy` and `{dataset}_labels_{train,test}.txt` |
| 3. Noun selection | `python selectNouns.py` | Training image embeddings and WordNet embeddings | `data/3_related_nouns_for_each_center/related_nouns_for_each_center_{dataset}.pkl` |
| 4. Prompt generation | `python promptGenerate.py` | Saved noun assignments; configured LLM API | `data/2_M_prompts/initialPrompts/DeepSeek-V3.2/{dataset}_K_prompts_for_each_center.npy` |
| 5. Evolutionary optimization | `python PromptOptimization.py` | Test image embeddings and labels; generated prompts | Clustering metrics printed to the terminal (no final result file is saved by the supplied script) |

### Configuration details

- The scripts are currently configured for **CIFAR-20** by default. Other paper datasets are **CIFAR-10, STL-10, DTD, ImageNet-10, and ImageNet-Dogs**; adjust the relevant settings in each script before use.
- `selectNouns.py` currently uses `cluster_num = 200` and `topK = 3` for CIFAR-20. Its implementation skips preliminary clusters to which no nouns are assigned, so the number of prompt groups can be smaller than 200.
- `promptGenerate.py` requests **five prompts per noun group** (`M = 5`) and batches up to **eight groups per API request**. Configure the model name, API endpoint, and credentials before running. **Never commit a real API key**.
- `PromptOptimization.py` uses **population size 20** and **10 evolutionary iterations** in the supplied configuration. It expects five prompts per group.
- On its first run, if `{dataset}_initial_promptEmbeddings.npy` is absent, `PromptOptimization.py` generates and saves this cache **then exits**. Run the script again to start evolutionary optimization.

### Data preparation

`dataLoader.py` can download CIFAR-10, CIFAR-100 (used for CIFAR-20), STL-10, and DTD through torchvision. **ImageNet-10 and ImageNet-Dogs require local `train/` and `val/` folders** under `data/4_datasets/`.

The uploaded code bundle does **not** contain full benchmark datasets or all intermediate embeddings. It includes a WordNet noun CSV and a limited CIFAR-20 prompt/noun example. You must prepare the remaining data and intermediate files to run other datasets.

## Implementation status

**Research-code snapshot — not yet a verified end-to-end, label-free reproduction release.** In the supplied `PromptOptimization.py`, ground-truth labels are used to align clustering probabilities inside fitness evaluation. This needs to be addressed and the results revalidated before describing the released implementation as fully unsupervised. The implementation of some other operations also differs from the manuscript equations.

## Citation

Please cite *Unsupervised Evolutionary Prompt Optimization for Image Clustering* by Wangwang Li, Na Chen, Qun Chen, Wenjie Liu, and Zhanhuai Li. Full bibliographic details can be added after publication.
