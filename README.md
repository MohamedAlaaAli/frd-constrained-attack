# Frd-Constrained Attack

On the Feasibility of Fréchet Radiomic Distance–Constrained Adversarial Examples in Medical Imaging: Methods and Trade-offs

# Abstract 

Adversarial attacks expose critical vulnerabilities in medical imaging AI models; yet, most
existing methods violate the textural and structural characteristics that define authentic 
medical images by disregarding the clinical and radiomic plausibility of the generated
perturbations. As a result, adversarial examples often violate the textural and structural
characteristics that define authentic medical images. In this study, we present the first
systematic investigation in the existence and feasibility of adversarial examples constrained
by the Fr´echet Radiomic Distance (FRD) a quantitative measure of radiomic similarity
capturing textural, structural, and statistical coherence between images. We formulate a
gradient-free, multi objective optimization framework based on Multi Objective Particle
Swarm Optimization (MOPSO) operating in the Discrete Cosine Transform (DCT) domain.
This framework jointly minimizes FRD and maximizes adversarial deviation, allowing a
principled exploration of the trade off between radiomic fidelity and adversarial strength
without requiring gradient access. Empirical evidence across multiple medical imaging
models demonstrates that enforcing strong FRD constraints (FRD ≤ 0.05) dramatically
reduces adversarial feasibility. Perturbations preserving radiomic fidelity consistently fail
to achieve meaningful adversarial deviation, suggesting that radiomic realism imposes an
intrinsic feasibility boundary on adversarial generation. These findings establish radiomic
consistency as a fundamental constraint on adversarial vulnerability, offering theoretical
and empirical insight toward the development of inherently robust and trustworthy medical imaging AI

---

# Data & Models

## Datasets

Our experiments are conducted on three publicly available dermatology datasets. Each dataset is partitioned into tasks with mutually exclusive class labels.

| Dataset               | Source                                                                                              | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **HAM10000 (HAM)**    | [Download](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)                                              | Dermoscopic images of 7 pigmented lesion classes.                                             |
| **Dermofit (DMF)**    | [Download](https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library)           | High-quality skin lesion images collected under standardised conditions with internal colour standards. |
| **Derm7pt (D7P)**     | [Download](https://derm.cs.sfu.ca/Welcome.html)                                                       | Dermoscopic dataset designed to follow the 7-point skin lesion malignancy checklist.          |


---

### Foundation Models

All adversarial examples were generated targeting the **OpenAI CLIP** model, then all embeddings models were used to compute adversarial distances.


| Model        | Source / Description                                                                                              |
|--------------|--------------------------------------------------------------------------------------------------------------------|
| **Derm**     | [Google Derm Foundation Model](https://huggingface.co/google/derm-foundation), trained on over 400 skin conditions. |
| **PanDerm**  | [PanDerm](https://github.com/SiyuanYan1/PanDerm), pretrained on millions of clinical and dermoscopic dermatology images. |
| **CLIP**     | [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14), pretrained on large-scale image-text pairs.  |

---

# Data Structure

To run the experiments, ensure your data is organized as follows:

```
project_root/
├── datasets/
│   ├── HAM/          # HAM10000 dataset root
│   ├── DERM7PT/      # Derm7pt dataset root (should contain meta/meta.csv and images/)
│   └── DMF/          # Dermofit dataset root
└── test_sets/
    ├── test_ids/
    │   ├── dmf_ids.csv
    │   ├── ham_ids.csv
    │   └── d7p_ids.csv
    └── test_images/  # Directory where processed test images will be saved
        ├── dmf/
        ├── ham/
        └── derm7pt/
```

---

# Usage

To run the FRD-constrained attack using MOPSO, use the `pso.py` script. You can specify the dataset and other parameters via command-line arguments.

## Running the Attack

```bash
python pso.py --dataset <dataset_name> [options]
```

### Arguments

- `--dataset`: The dataset to attack. Choices: `dmf`, `ham`, `derm7pt`. Default: `dmf`.
- `--num_samples`: Number of samples to attack. Default: `243`.
- `--output_dir`: Directory to save results. Default: `results/<dataset>/pso`.
- `--swarm_size`: Size of the particle swarm. Default: `50`.
- `--iters`: Number of iterations for the optimization. Default: `80`.
- `--device`: Device to use (`cuda` or `cpu`). Default: auto-detect.

### Examples

Run on DMF dataset with default settings:
```bash
python pso.py --dataset dmf
```

Run on HAM dataset with 100 samples and custom output directory:
```bash
python pso.py --dataset ham --num_samples 100 --output_dir results/my_ham_experiment
```

## Results

The results will be saved in the specified `output_dir`.
- `summary_all.csv`: Contains aggregated statistics for all processed samples.
- `global_pareto.png`: A plot showing the global Pareto front (FRD vs Adversarial Distance).
- Individual sample directories (e.g., `sample_0/`) containing:
    - `progress.csv`: Per-iteration progress of the optimization.
    - `summary.json`: Final statistics for the sample.
    - Generated adversarial images (e.g., `iter_final_top0_...png`).

---

# Acknowledgement
## Original FRD-paper
```bibtex
@article{konzosuala_frd2025,
      title={Fr\'echet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets}, 
      author = {Konz, Nicholas and Osuala, Richard and Verma, Preeti and Chen, Yuwen and Gu, Hanxue and Dong, Haoyu and Chen, Yaqian and Marshall, Andrew and Garrucho, Lidia and Kushibar, Kaisar and Lang, Daniel M. and Kim, Gene S. and Grimm, Lars J. and Lewin, John M. and Duncan, James S. and Schnabel, Julia A. and Diaz, Oliver and Lekadir, Karim and Mazurowski, Maciej A.},
      year={2025},
      eprint={2412.01496},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.01496}, 
}
```
