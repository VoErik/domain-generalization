
# Data Augmentation for Domain Generalization

---

Short description of what we do.

## Installation

---




+ add requirements

## Components

---

* `domgen/augmentor.py`: Definition of augmentation strategies that are handled by an `Augmentor` class during training.
* `domgen/trainer.py`: Handling of the training process based on a configuration file.
* `domgen/tuning.py`: Classes for hyperparameter and augmentation tuning.
* `domgen/domain_dataset.py`: Definition of a dataset class to handle specific domain datasets.

## Quickstart

---

### Setup

From `root` run:
```console
python -m pip install -e .
```

### Using the Trainer

To run experiments, simply instatiate an instance of the `DomGenTrainer`, pass a configuration by either:
* defining a `config.yaml` (`.json`) file and passing it via the `--config` flag when running `train.py` or
* defining a `dataclass` or similar and passing it to the trainer.


```python
from argparse import ArgumentParser
from domgen import DomGenTrainer

parser = ArgumentParser()
args = parser.parse_args() # args must be a namespace object / dataclass

trainer = DomGenTrainer(args)
trainer.fit()
```

After training, the results of the experiments can be saved by calling

```python
trainer.save_metrics(trainer.metrics, experiment_path)
trainer.save_config(f'{experiment_path}/trainer_config.json')
```
which save the results of the training to file.

⚡ *Note, that saving the trainer config requires it to be serializable. This needs to be handled individually.* ⚡

Additionally, we provide functions to plot the results:

```python
from domgen.utils import plot_accuracies, plot_training_curves

# experiment_path is the path to the directory where 
# the results of the run were saved to
experiment_path = f"{args.log_dir}/{args.experiment}"

plot_accuracies(root_path=experiment_path, save=True, show=False)
plot_training_curves(base_dir=experiment_path, show=False)
```

### Available Augmentation Strategies

We provide a range of augmentation strategies:

* `no_augment`: This is the identity transformation.
* `mixup`: Using the implementation of PyTorch [here](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.MixUp.html).
* `cutmix`: Using the implementation of PyTorch [here](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.CutMix.html).
* `augmix`: Using the implementation of PyTorch [here](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.AugMix.html).
* `randaugment`: Using the implementation of PyTorch [here](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RandAugment.html).
* `mixstyle`: Using our own implementation based on [Zhou et al. 2021](https://arxiv.org/abs/2107.02053).
* `pacs_custom`: A set of handcrafted image-level augmentations.
* `medmnistc`: We provide a wrapper for [di Salvo et al. 2024](https://arxiv.org/pdf/2406.17536).

For both `pacs_custom` and `medmnistc` you need to specify an additional field in the config `aug_dict`, that specifies which of the augmentations you want to use.

You can extend this by adding new augmentation strategies. See [Extending the Code](#Extending the Code).

### Hyperparameters

We provide configuration files for our models with the hyperparameters that worked best. Alternatively, you can run your own trials.
To run a trial run for one model-config simply run:

```console
python tune_config --data_config [PATH/TO/CONFIG.yaml] --hp_config [PATH/TO/CONFIG.yaml]
```
Additionally, we provide scripts for tuning all models of a certain class as well as running a full sweep on all configs.


## Data

---

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

## Extending the Code

---

### Adding datasets

You can easily extend this repository by adding additional datasets. For that, you need to make sure that the dataset is placed in the `datasets` directory
and follows the structure as shown (subdirectories for each domain containing the class directories):

```console
├── dataset_root
│   ├── domain1
│   │   ├── cls1
│   │   ├── cls2
│   │   ├── ...
│   ├── domain2
│   │   ├── cls1
│   │   ├── cls2
│   │   ├── ...
...

```

### Adding Augmentation Strategies

tbc.



### Resources

Zamanitajeddin et al. (2024): [Benchmarking Domain Generalization Algorithms in Computational Pathology](https://arxiv.org/html/2409.17063v1)

Aminbeidokhti et al. (2023): [Domain Generalization by Rejecting Extreme Augmentations](https://arxiv.org/pdf/2310.06670)
