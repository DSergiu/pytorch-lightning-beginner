<h1 align="center">
  <b>Pytorch Lightning Beginner</b><br>
</h1>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.7-ff55aa.svg" />
    </a>
    <a href="https://pytorch-lightning.readthedocs.io/en/stable/">
        <img src="https://img.shields.io/pypi/v/pytorch-lightning?label=Pytorch%20Lightning" />
    </a>
    <a href= "https://github.com/DSergiu/pytorch-lightning-beginner/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" />
    </a>
</p>


Showcase different ML techniques using Pytorch Lightning inspired by [pytorch-beginner](https://github.com/L1aoXingyu/pytorch-beginner).

The repository is split into `chapters` starting with basic linear regression and ending with more complex networks (e.g. RNNs, GANs). 
There is a jupyter notebook within each chapter where one can experiment on training and usage of the trained models.

## How to use?

1. Clone repository and install dependencies
    * `git clone https://github.com/DSergiu/pytorch-lightning-beginner`
    * `cd pytorch-lightning-beginner`
    * `pip install -r requirements.txt`
2. Run jupyter notebook
    * `jupyter notebook`
3. Open a notebook chapter and experiment (e.g. )
    * e.g. open `01-Linear-Regression/Notebook.ipynb`

## Useful training arguments

When training you can experiment with following arguments. For a full list see [Pytorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html).

|Argument|Description|
|----------|-------------|
|[`--help`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html)| see list of arguments |
|[`--fast_dev_run=True`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#fast-dev-run)| fast run of training |
|[`--max_epochs=20`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#max-epochs)| run 20 epochs of training data set |
|[`--gpus=2`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#gpus)| train using 2 gpus |
|[`--deterministic=True`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#reproducibility)| training always produces same output given same input |

## Prerequisites
You must be familiar with Python, Pytorch and Pytorch Lightning.

To understand the code please see official [Pytorch Lighning Docs](https://pytorch-lightning.readthedocs.io/en/stable/).

## Requirements
* jupyter-notebook
* python 3.7
* pytorch 1.0.0+

## TODO
Following types of neural networks are planned to be added:
* Recurrent Neural Network (LSTM, GRU)
* Language Models
* Generative Adversarial Network
