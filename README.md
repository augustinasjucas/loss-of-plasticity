# Layerwise Perspective into Continual Backpropagation

## Contents

- [Overview](#overview)
- [Repository Contents](#repository-contents)
- [System Requirements](#system-requirements)
- [Installation Guide (Linux)](#installation-guide-linux)

## Overview
This repository contains the code for running experiments and analyzing data for the Bachelor thesis titled *"Layerwise Perspective into Continual Backpropagation: Replacing the First Layer is All You Need"*, authored by **Augustinas Jučas**.

### Thesis Information
- **Institution:** Delft University of Technology
- **Study Programme:** Bachelor of Computer Science and Engineering
- **Supervisor:** Laurens Engwegen
- **Responsible Professor:**  Wendelin Böhmer
- **Examiner:** Megha Khosla

This repository is a fork and and extension of Dohare et al.'s work: [Loss of Plasticity GitHub Repository](https://github.com/shibhansh/loss-of-plasticity).

## Repository Contents
- [lop/utils](./lop/utils): Some utility functions used throughout the repository.
- [lop/algos](./lop/algos): All the algorithms used in the paper.
- [lop/nets](./lop/nets): The network architectures used in the paper.
- [lop/bit_flipping](./lop/bit_flipping): Core for the Bit-Flipping problem's experiments. Also includes parameters and data analysis scripts.
- [lop/permuted_mnist](./lop/permuted_mnist): Code for the Continual Permuted MNIST problem. Also includes raw data, parameters and data analysis scripts.

The latter two directories contain README files of their own, each explaining how to reproduce the respective experiments.

## System Requirements

To be able to run any single experimental instance, 3GB of RAM and an single CPU core are enough. However, having more resources would allow running experiments in parallel.

All experiments should definitely run on Ubuntu 22.04, however, we expect them to work on any properly installed Python environment, denoted below.

## Installation Guide (Linux)

Create a virtual environment
```sh
mkdir ~/envs
virtualenv --no-download --python=/usr/bin/python3.8 ~/envs/plast
source ~/envs/plast/bin/activate
pip3 install --no-index --upgrade pip
```

Install the requirements:
```sh
pip3 install -r requirements.txt
pip3 install -e .
```

Add this lines in your `~/.zshrc` or `~/.bashrc`
```sh
source ~/envs/plast/bin/activate
```
