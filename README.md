<h1 align="center">
  <b>Pytorch Lightning Beginner</b><br>
</h1>

<p align="center">
   <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-ff55aa.svg" />
   </a>
   <a href="https://pytorch-lightning.readthedocs.io/en/stable/">
      <img src="https://img.shields.io/badge/Pytorch%20Lightning-v1.8.3.post1-blue.svg" />
   </a>
   <a href= "https://github.com/DSergiu/pytorch-lightning-beginner/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-blue.svg" />
   </a>
</p>


Showcase different ML techniques using Pytorch Lightning inspired by [pytorch-beginner](https://github.com/L1aoXingyu/pytorch-beginner).

The repository is split into `chapters` starting with basic linear regression and ending with more complex networks (e.g. RNNs, GANs). 
There is a jupyter notebook within each chapter where one can experiment on training and usage of the trained models.

## How to use?

1. Clone repo
    * `git clone https://github.com/DSergiu/pytorch-lightning-beginner`
    * `cd pytorch-lightning-beginner`
2. Create and activate virtual env
    * `python3 -m venv venv && ./venv/Scripts/activate`
3. Install dependencies
    * `pip install -r requirements.txt`
4. Start jupyter notebook
    * `jupyter notebook`
5. Open a notebook chapter and experiment (e.g. )
    * e.g. open `01-Linear-Regression/Notebook.ipynb`

## Useful training arguments

When training you can experiment with following arguments. For a full list see [Pytorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html).

| Argument               | Description                                           |
|------------------------|-------------------------------------------------------|
| `--help`               | see list of arguments                                 |
| `--fast_dev_run=True`  | fast run of training                                  |
| `--max_epochs=20`      | run 20 epochs of training data set                    |
| `--accelerator=cuda`   | train on cuda GPU                                     |
| `--devices=2`          | train using 2 devices                                 |
| `--deterministic=True` | training always produces same output given same input |

## Prerequisites

You must be familiar with Python, Pytorch and Pytorch Lightning.

To understand the code please see official [Pytorch Lighning Docs](https://pytorch-lightning.readthedocs.io/en/stable/).

## Requirements

* python 3.8
