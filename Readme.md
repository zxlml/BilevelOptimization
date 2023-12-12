# Stability-based Generalization Assessment for Stochastic Bilevel Optimization

Stochastic bilevel optimization (SBO) has been integrated into many machine learning paradigms recently including hyperparameter optimization, meta learning,  reinforcement learning, etc. Along with the wide range of applications, there have been abundant studies on concerning  the computing  behaviors of SBO. However, the generalization guarantees of SBO methods are far less understood from the lens of statistical learning theory. In this paper, we provide a systematical generalization analysis of  the first-order gradient-based bilevel optimization methods. Firstly, we establish the quantitative connections between the on-average argument stability and the generalization gap of SBO methods. Then, we derive the upper bounds of on-average argument stability for single timescale stochastic gradient descent (SGD) and two timescale SGD, where three settings (nonconvex-nonconvex (NC-NC), convex-convex (C-C) and strongly-convex-strongly-convex (SC-SC)) are considered respectively. Experimental analysis validates our theoretical findings. Compared with the previous algorithmic stability analysis,  our  results do not require the re-initialization of the inner-level parameters before each iteration and are suit for more general objective functions.  

# Code & Data Acknowledgement

## Hyperparameter Optimization

As for hyperparameter optimization task, we employed the codes from [1] (https://github.com/baofff/stability_ho) with several modifications (e.g., remove the re-initialization operation).

## Meta Learning

(2) As for meta learning task, we employed the codes from [2] (https://github.com/JunjieYang97/stocBiO) .



# Data Source

# MNIST data set for data cleaning 

MNIST data [3] can be downloaded from the Python library "torch.utils.data"

# Omniglot data set for one-shot learning

Omniglot data [4] can be downloaded from https://github.com/brendenlake/omniglot



## Reference

[1] Stability and generalization of bilevel programming in hyperparameter optimization

```
@article{bao2021stability,
  title={Stability and generalization of bilevel programming in hyperparameter optimization},
  author={Bao, Fan and Wu, Guoqiang and Li, Chongxuan and Zhu, Jun and Zhang, Bo},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={4529--4541},
  year={2021}}
```

[2] Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms

```
@inproceedings{ji2021bilevel,
	author = {Ji, Kaiyi and Yang, Junjie and Liang, Yingbin},
	title = {Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms},
	booktitle={International Conference on Machine Learning (ICML)},
	year = {2021}}
```

[3]  MNIST dataset

```
@article{lecun1998mnist,
  title={The MNIST database of handwritten digits},
  author={LeCun, Yann},
  journal={http://yann. lecun. com/exdb/mnist/},
  year={1998}}
```

[4]  Omniglot dataset

```
@article{
author = {Brenden M. Lake  and Ruslan Salakhutdinov  and Joshua B. Tenenbaum },
title = {Human-level concept learning through probabilistic program induction},
journal = {Science},
volume = {350},
number = {6266},
pages = {1332-1338},
year = {2015}}
```

