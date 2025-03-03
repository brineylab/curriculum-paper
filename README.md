[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14661302.svg)](https://doi.org/10.5281/zenodo.14661302)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# Curriculum learning for antibody language models

We introduce a method of curriculum learning for antibody language models (AbLMs), as an approach for pre-training AbLMs with both unpaired and natively paired sequence data. We compare this method to other pre-training approachs, such as finetuning and a constant mix. We also train a 650M parameter model, CurrAb, using our curriculum implementation. This repository contains the code used to train and evaluate the models presented in the paper.

* [**Curriculum training mods**](curriculum-mods/): includes the configs, modified dataset, and trainer callback used to train the curriculum models
* [**55M-param model training**](model-training_55M/): training code for 55M parameter models in Figures 2-4
* [**650M-param model training**](model-training_650M): training code for 650M parameter models in Figure 5 and Table S4, including CurrAb
* [**Model eval**](model-eval/): evaluation code used in the paper, including inference and specificity classification

Training datasets can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.14661302) and model weights for CurrAb are avaliable on [Hugging Face](https://huggingface.co/brineylab/CurrAb).

### how should I cite this?
The curriculum learning paper has been published as a [preprint on biorxiv](https://www.biorxiv.org/content/10.1101/2025.02.27.640641v1), and can be cited as:

```
Burbach, S. M., & Briney, B. (2025). A curriculum learning approach to training antibody language models.
bioRxiv (p. 2025.02.27.640641). https://doi.org/10.1101/2025.02.27.640641

```

The current version of the dataset (v2025.02.25) can be cited as:

```
Burbach, S.M., & Briney, B. (2025). Curriculum learning for antibody language models (v2025.02.25) [Data set]. 
Zenodo. https://doi.org/10.5281/zenodo.14661302
```
