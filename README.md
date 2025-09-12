[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14661302.svg)](https://doi.org/10.5281/zenodo.14661302)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# Curriculum learning for antibody language models

We introduce a method of curriculum learning for antibody language models (AbLMs), as an approach for pre-training AbLMs with both unpaired and natively paired sequence data. We compare this method to other pre-training approaches, such as finetuning and a constant mix. We also train a 650M parameter model, CurrAb, using our curriculum implementation. This repository contains the code used to train and evaluate the models presented in the paper.

* [**Curriculum training mods**](curriculum-mods/): includes the configs, modified dataset, and trainer callback used to train the curriculum models
* [**55M-param model training**](model-training_55M/): training code for 55M parameter models
* [**650M-param model training**](model-training_650M/): training code for 650M parameter models, including CurrAb
* [**Model eval**](model-eval/): evaluation code used in the paper, including inference and specificity classification
* [**Example using CurrAb**](CurrAb-example.ipynb): example using CurrAb to infill masked positions

Training datasets can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.14661302) and model weights are available on [Hugging Face](https://huggingface.co/collections/brineylab/curriculum-paper-685b08a4b6986df7c5a5e3c4).

### how should I cite this?
The curriculum learning paper has been published in [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1013473), and can be cited as:

```
Burbach SM, Briney B (2025) A curriculum learning approach to training antibody language models.
PLoS Comput Biol 21(9): e1013473. https://doi.org/10.1371/journal.pcbi.1013473

```

The current version of the dataset (v2025.02.25) can be cited as:

```
Burbach, S.M., & Briney, B. (2025). Curriculum learning for antibody language models (v2025.02.25) [Data set]. 
Zenodo. https://doi.org/10.5281/zenodo.14661302
```
