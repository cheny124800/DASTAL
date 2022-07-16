## DASTAL
We propose a diversity-aware semantic transformation active learning, or DAST-AL framework, that looks ahead the effect of ISDA in the process of acquisition. Specifically, DAST-AL exploits expected partial model change maximization (EPMCM) to consider selected samples' potential contribution of the diversity to the labeled set by leveraging the semantic transformation within ISDA when selecting the unlabeled samples.


## Getting started
### Install

1. Create an Anaconda environment：

  torch >= 1.1.0

  numpy >= 1.16.2

2. Dataset
    cifar10, cifar100
    
    
### Train and test
- Train
    ~~~
    python main_DASTAL.py
    ~~~


## Acknowledgements
This code is partly based on the open-source implementations from [ISDA](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks) and [Agreement-Discrepancy-Selection](https://github.com/fumengying19/AAAI21-ADS).
