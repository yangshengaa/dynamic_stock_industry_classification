# Dynamic Stock Industrial Classification

Use graph-based analysis to re-classify stocks and experiment different re-classification methodologies to improve Markowitz portfolio optimization performance in the low-frequency quantitative trading context.

Note that for strategy confidentiality, many files are hidden.

## Module Breakdown

This project contains the following five modules:

- [factor generation](src/factor_generation): compute and store factors alpha factors and risk factors for low-frequency trading;
- [backtest](src/backtest): low-frequency backtest framework;
- [factor combination](src/factor_combination): combine factors using ML models;
- [portfolio optimization](src/portfolio_optimization): Markowitz portfolio optimization, with turnover, industrial exposure, style exposure, and various other constraints.
- [graph clustering](src/graph_cluster): experiment different graph-based clustering on stocks.

## Data

China A-Share stocks, the corresponding major index data (sz50, hs300, zz500, zz1000), and the member stock weights from 20150101 to 20211231, provided by Shanghai Probability Quantitative Investment.

## Quick Start

It's very easy to use this platform!

Tips:

- run each module at a time;
- change config for corresponding module in respective files (file location indicated inside [run.py](run.py)).

To run each module, in current directory:

- factor generation: `python run.py gen`
- backtest: `python run.py backtest`
- factor combination: `python run.py comb`
- portfolio optimization: `python run.py opt`
- graph clustering: `python run.py cluster`

## Acknowledgement

Special thanks to coworkers and my best friends at Shanghai Probability Quantitative Investment: Beilei Xu, Zhongyuan Wang, Zhenghang Xie, Cong Chen, Yihao Zhou, Weilin Chen, Yuhan Tao, Wan Zheng, and many others. This project would be impossible without their data, insights, and experiences.
