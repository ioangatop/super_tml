# SuperTML

Author: Ioannis Gatopoulos, 2020

<p align="center">
  <img src="readme_imgs/super_tml.png" width="500" />
</p>


## Description
PyTorch implementation of research paper:

_SuperTML: Two-Dimensional Word Embedding for the Precognition on Structured Tabular Data_


_Abstrack_: Projects tabular data to images creating 2-dimentional embenddings. Then, with the use of a pretrained model, it is capable of performing regression and classification tasks.

•   [Code](src/)   •   [Paper](https://arxiv.org/pdf/1903.06246.pdf)   •   [Blog-Post (soon)]()


## Results
<p align="center">
  <img src="readme_imgs/results.png" width="600" />
</p>


## Required Python packages
Install all the dependencies:
```bash
pip3 install -r requirements.txt
```

## Run
```
cd super_tml
python main.py --dataset iris --model densenet121
```

## References
    - Baohua Sun.
    SuperTML: Two-Dimensional Word Embedding for the Precognition on Structured Tabular Data.
    CVPR Workshop Paper, 2019.
