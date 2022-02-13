---
title: "Dynamic Stock Industry Classification"
toc: yes
---

## Background

Stocks are classified into different sectors (Energy, Finance, Health Care, etc), and stocks within the same sectors are assumed to have similar behaviors (movement patterns and risk profiles). Fund managers worldwide demand a precise classification to control portfolio sector exposures and thus minimize risks brought by some specific sectors. This could be considered a sector-level diversification.  

The most widely used industry classifications are China CITIC (中信) and SWS Research (申万宏源) for China A-share. They provide a professional guideline for a long-term stock industry classification. However, the classification is fixed and fails to capture short-term correlations between stocks in different sectors, and thus fails to embody short-term co-movement risks between conventionally unrelated stocks. For example, company A in finance sector and company B in energy sector are typically considered uncorrelated. Due to a recent announcement of cooperation, their stock prices started to behave similarly. This particular risk could hardly be hedged against if the fund manager use a fixed industry classification scheme.  

Therefore, a dynamic industry classification is much more recommended for institutional traders, especially hedge fund portfolio managers on low-frequency trading strategies (change stock holdings each day, for instance).  

<center>
<figure>
<img src="out/graph_fig/cluster_demo_pmfg.png" alt="Trulli" style="width:70%">
<figcaption align = "center"><b>Fig 1: Trained zz1000 Member Stock Classification (2021/01/01 - 2021/12/31) </b></figcaption>
</figure>
</center>

## Graph Formulation

To re-classify stocks from stock data, we believe that graph helps filter information and compare otherwise hard-to-contrast variables (price movements in this case) to obtain hidden embeddings of industry information. In a nutshell, two stocks are connected if they demonstrate a strong correlation over the given observation time period, and by that connectivity we may partition the graph and obtain communities.

### Build Graph from Financial Data

We would like to build a graph whose nodes are stocks and edges are indicators of connectivity. Suppose there are $N$ tradable assets and $T$ days for observation, we take the time-series correlation among stocks as a criteria to add edges.

To compute the time-series correlation, suppose $s_{i,t}$ is the (close) price of asset $i$ at time $t \in \{1, ..., T\}$, then the daily return is $r_{i, t} = \frac{s_{i, t} - s_{i, t - 1}}{s_{i, t - 1}}$ ($t$ starts from 2, which means there are only $T - 1$ returns). Then for any $i, j$, the time-series correlation is thus given by 
$$\rho_{ij} = \frac{\sum_{t=2}^T (r_{i, t} - \bar{r}_i)(r_{j, t} - \bar{r}_j) }{\sqrt{[\sum_{t=2}^T (r_{i, t} - \bar{r}_i)^2] [\sum_{t=2}^T (r_{j, t} - \bar{r}_j)^2]}}$$
where $\bar{r}_i = \frac{\sum_{t = 2}^T r_{i, t}}{T - 1}$. This could be considered as the "weight" of the edge between stock $i$ and stock $j$. One sometimes need to convert weights to distance between two nodes, and a naive form is give by 
$$d_{ij} = \sqrt{2 (1 - \rho_{ij})}$$

Given the similarity measures (correlation) and distance measures, we may build graphs by using the following methods:

- **Asset Graph (AG)**: connect if $|\rho_{ij}|$ is beyond a pre-defined threshold;  
- **Minimum Spanning Tree (MST)**: sort all $\rho_{ij}$ in a descending order, add the edge if after addition the graph is still a forest or a tree (Kruskal's Algorithm);  
- **Planar Maximally Filter Graph (PMFG)**: simiilar to MST, but add edge if after addition the graph is still planar;
- **Random Matrix Theory (RMT)**: select information from the correlation matrix and feed back to the previous three models as a refinement.

In this project we use all four types in our experiment.

### Community Detection from Constructed Graphs

To control the number of industry, we pick algorithms that help generate a prescribed number of clusters. The following are implemented:

- **Spectral Clustering**
- **Average Linkage Clustering**
- **Node2Vec + KMeans**: conduct KMeans on Node2Vec embeddings;
- **Sub2Vec + KMeans**: conduct KMeans on Sub2Vec embeddings.

### Graph Evaluation

To evaluate if the re-constructed classification is "good", we go through the entire low-frequency stock picking pipeline and plug in new industry information in the final step -- Markowitz Portfolio Optimization -- to see if there is a performance gain in our strategy.

We focus on the following four metrics to measure performance:

- excess return: the excess return of the strategy with respect to the index / market;
- max drawdowns: max decrease of the portfolio in value;
- turnover: measure the rate of invested stocks being replaced by new ones;
- AlphaSharpeRatio: return / volatility, measure the ability of maximizing returns over risk.

The dynamic property is done by a rolling-based train test schemed outlined as follows: we train the graph using $T_{train} = 240$ days and test the performance of the graph in the following $T_{test} = 40$ days. Then we move forward $T_{test}$ days to retrain the graph. Note that the test periods are not overlapping, and the train test periods are the same in the factor combination (machine learning) part of the low-frequency stock picking paradigm. We look at the metrics of the successive testing periods in our portfolio.


<center>
<figure>
<img src="report/rolling_test.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig 2: Rolling Evaluation Paradigm </b></figcaption>
</figure>
</center>

## Experiment

### 4.1 Data

Provided by Shanghai Probability Quantitative Investment, this is a dataset of day-level A-share stock information.

In this project, we will focus on a particular stock pool named zz1000 (中证1000) favored by many investors. This is a pool of 1000 mid-size market cap stocks, and the pool replace stocks every 6 months. The following stats on zz1000 is taking the union of all stocks appeared in this pool in history.

## Reference
