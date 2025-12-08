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


# Citation

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
