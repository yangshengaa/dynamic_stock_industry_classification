# Dynamic Stock Industrial Classification

Use graph-based analysis to re-classify stocks and experiment different re-classification methodologies to improve Markowitz portfolio optimization performance in the low-frequency quantitative trading context.

Note that for strategy confidentiality, many files are hidden.

To accommodate speedy development, the current code structure simplicity is sacrificed. This will be addressed in later versions.

## Module Breakdown

This project contains the following five modules:

- [factor generation](src/factor_generation): compute and store factors alpha factors and risk factors for low-frequency trading;
- [backtest](src/backtest): low-frequency backtest framework (both factors and signals). Factors have continuous values on each cross section whereas signals have only -1, 0, and 1 overall;
- [factor combination](src/factor_combination): combine factors using ML models;
- [portfolio optimization](src/portfolio_optimization): Markowitz portfolio optimization, with turnover, industrial exposure, style exposure, and various other constraints.
- [graph clustering](src/graph_cluster): experiment different graph-based clustering on stocks.

## Data

China A-Share stocks, the corresponding major index data (sz50, hs300, zz500, zz1000), and the member stock weights from 20150101 to 20211231, provided by Shanghai Probability Quantitative Investment.

## Experiment Results

**Stock Pool**: zz1000 member stocks  
**Benchmark**: zz1000 index  
**Time Period**: 20170101 - 20211231  

| Model | AlphaReturn (cumsum) | AlphaSharpe | AlphaDrawdown | Turnover |
| ----- | :---------------------: | :----------: | :-----------: | :------: |
| LinearRegressor | 82.55 | 2.06 | -19.89 | 1.0 |
| LgbmRegressor | 150.56 | 3.47 | -11.14 | 1.17 |

TODO: add stats after portfolio opt;  
TODO: add graph improved model stats;

## Quick Start

It's very easy to use this platform!

Tips:

- run each module at a time, and run the following command sequentially;
- change config for corresponding module in respective files (file location indicated inside [run.py](run.py)).

To run each module, in current directory:

Factor Generation:

- factor generation: `python run.py gen`

Backtest:

- backtest factor: `python run.py backtest_factor`
- backtest signal: `python run.py backtest_signal`

Factor Combination:

- factor combination: `python run.py comb`

Portfolio Optimization:

- generate factor returns: `python run.py opt_fac_ret`
- estimate covariance matrices: `python run.py opt_cov_est`
- adjust weight: `python run.py opt_weight`

Graph Clustering:

- train graph clustering: `python run.py cluster_train`

To run each submodules, in current directory:

- generate pairs factors: `python run.py pairs`
- generate risk factors: `python run.py gen_risk`

Currently risk attribution module is very slow and suboptimal. To be addressed later.

## Acknowledgement

Special thanks to coworkers and my best friends at Shanghai Probability Quantitative Investment: Beilei Xu, Zhongyuan Wang, Zhenghang Xie, Cong Chen, Yihao Zhou, Weilin Chen, Yuhan Tao, Wan Zheng, and many others. This project would be impossible without their data, insights, and experiences.

## For Developer

Log known issues here:

- signals given by factor test could not give the same alpha returns (slightly less) as in signal test
  - examine output holding stats
- plain risk attribution
