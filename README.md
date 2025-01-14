
## Installation
From `root` run:
```console
python -m pip install -e .
```

## Hyperparameters

We ran a basic hyperparameter search for each leave-out split using `raytune`. For each trial run we sampled 30 times and trained for 10 (pretrained) or 100 (new initialization) epochs. Generally, no clear inter-domain agreement on **a** best hyperparameter configuration could be found. Instead, we opted for the agreed upon set with the highest validation accuracy each, with no validation accuracy < 92%.

| Model            | LR     | Batch Size | Weight Decay | Optimizer | Betas       | Eps  |
|------------------|--------|------------|--------------|-----------|-------------|------|
| Resnet18 (pt)    | 1e-3   | 64         | 0.1          | AdamW     | 0.9, 0.999  | 1e-8 |
| Resnet34 (pt)    | 1e-3   | 64         | 0.1          | AdamW     | 0.9, 0.999  | 1e-8 |
| Resnet50 (pt)    | 1e-3   | 64         | 0.1          | AdamW     | 0.9, 0.999  | 1e-8 |
| Resnet18         | 1e-3   | 64         | 0.1          | AdamW     | 0.9, 0.999  | 1e-8 |
| Resnet34 (pt)    | 1e-3   | 64         | 0.1          | AdamW     | 0.9, 0.999  | 1e-8 |
| Resnet50 (pt)    | 1e-3   | 64         | 0.1          | AdamW     | 0.9, 0.999  | 1e-8 |
| Densenet121 (pt) | 1e-3   | 48         | 0            | Adam      | 0.9, 0.999  | 1e-8 |
| Densenet161 (pt) | 1e-3   | 32         | 0.003        | AdamW     | 0.9, 0.95   | 1e-8 |
| Densenet121      | 1e-3   | 48         | 0            | Adam      | 0.99, 0.999 | 1e-8 | 
| Densenet161      | 1e-3   | 32         | 0            | Adam      | 0.99, 0.999 | 1e-8 | 

*(pt) = pre-trained (ImageNet weights).*

For the criterion we chose `CrossEntropy` for all runs. 

To run a trial run for one model-config simply run:

```console
python tune_config --data_config [PATH/TO/CONFIG.yaml] --hp_config [PATH/TO/CONFIG.yaml]
```

To run a trial sweep for all models of a certain class (e.g. pretrained `resnet18` -> `resnet18-pretrained`), run:

```console
sh run_trials.sh [modelclass]
```
on Unix-based systems or

```console
run_trials.bat [modelclass]
```
on Windows. For a full sweep of all models, run:

```console
hp_full_sweep.ps1
```

This will train a total of (num_models * num_domains * num_samples) => 8 * 4 * 30 = 960 models.
## Data

We ran our experiments on two common domain generalization benchmarks: `PACS` and `Camelyon17`.
For convenience, we provide a utility script, that downloads both datasets and prepares the necessary directory structure.

To use the script, run `download_data.py`:
```console
python download_data.py --all
```
This loads both datasets into the `datasets` directory. To get only of the sets, simply specify either `--download_pacs` 
or `--download_camelyon17`.
The download directory can be changed via the `--datadir` flag.

Alternatively, you can manually load the datasets.

### PACS
The PACS dataset is a benchmark for domain generalization tasks. It consists of images from four domains: Photo, 
Art painting, Cartoon, and Sketch. The dataset includes seven object classes: Dog, Elephant, Giraffe, Guitar, Horse, 
House, and Person, making it suitable for evaluating models across diverse styles and distributions.

<figure>
  <img src="imgs/pacs.png" alt="Camelyon" style="width:100%">
  <figcaption>Fig.2 - Snippet from the PACS dataset. The images are 224x224 pixels and span seven classes over four domains.</figcaption>
</figure>

```tex
@misc{li2017deeperbroaderartierdomain,
      title={Deeper, Broader and Artier Domain Generalization}, 
      author={Da Li and Yongxin Yang and Yi-Zhe Song and Timothy M. Hospedales},
      year={2017},
      eprint={1710.03077},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1710.03077}, 
}
```

### Camelyon17
The Camelyon17 dataset is a benchmark for evaluating domain generalization in histopathology image analysis. It contains
images from five medical centers in the Netherlands, capturing inter-center variability (i.e. introduced by different
scanners). The dataset is designed for tumor classification tasks and assessing the robustness of machine learning models
in healthcare applications.

We use the patch-based variant of the dataset (Bandhi 2018). The data can be accessed via the 
[`WILDS` ecosystem](https://wilds.stanford.edu/datasets/#camelyon17) or downloaded directly from 
[here](https://camelyon17.grand-challenge.org/Data/).


<figure>
  <img src="imgs/camelyon.png" alt="Camelyon" style="width:100%">
  <figcaption>Fig.2 - Snippet from the Camelyon17 dataset. The images show 96x96 pixel patches from different patients lymph nodes from five different medical centers in the Netherlands.</figcaption>
</figure>



```tex
@article{bandi2018detection,
title={From detection of individual metastases to classification of lymph node status at the patient level: the CAMELYON17 challenge},
author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
journal={IEEE Transactions on Medical Imaging},
year={2018},
publisher={IEEE}
}
```
### Adding datasets

You can easily extend this repository by adding additional datasets. For that, you need to make sure that the dataset is placed in the `datasets` directory
and follows the structure as shown (subdirectories for each domain containing the class directories):

```console
├── PACS
│   ├── art_painting
│   │   ├── dog
│   │   ├── elephant
│   │   ├── giraffe
│   │   ├── guitar
│   │   ├── horse
│   │   ├── house
│   │   └── person
│   ├── cartoon
│   │   ├── dog
│   │   ├── elephant
│   │   ├── giraffe
│   │   ├── guitar
│   │   ├── horse
│   │   ├── house
│   │   └── person
...

```

### Resources

Zamanitajeddin et al. (2024): [Benchmarking Domain Generalization Algorithms in Computational Pathology](https://arxiv.org/html/2409.17063v1)

Aminbeidokhti et al. (2023): [Domain Generalization by Rejecting Extreme Augmentations](https://arxiv.org/pdf/2310.06670)
