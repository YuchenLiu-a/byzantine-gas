# README

This repository provides code of ICML'23 paper 

[Byzantine-Robust Learning on Heterogeneous Data via Gradient Splitting]: https://arxiv.org/abs/2302.06079

## Set up

Install conda environment

```sh
conda env create -f environment.yml
conda activate gas
```

## Running code

```sh
python main.py
```

change `config` in `main.py` to test GAS in different settings.

## Reference

```tex
@misc{liu2023byzantinerobust,
      title={Byzantine-Robust Learning on Heterogeneous Data via Gradient Splitting}, 
      author={Yuchen Liu and Chen Chen and Lingjuan Lyu and Fangzhao Wu and Sai Wu and Gang Chen},
      year={2023},
      eprint={2302.06079},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

