# PortfolioWeightOptimizer

FactorReturnGenerator: Compute Factor Returns and Idiosyncratic returns 计算因子收益率和特质收益率

CovMatrixEstimator: Estimate covariance matrix for factors and idio 计算因子收益率和特质收益率协方差矩阵

WeightOptimizer: optimize weight 计算优化持仓权重

TODO: 补充readme

## Running Instructions

To accommodate different trained industry, one may pass argument in command line as well. If no argument passed, read arguments from config. One may set up a bash file to estimate different industry in batch.

```bash
# run MST spectral clustering factor return estimates
python run.py opt_fac_ret --use_dynamic_ind True --dynamic_ind_name zz1000_10_MST_0_spectral_ 

# run MST spectral clustering covariance estimation 
python run.py opt_cov_est --use_dynamic_ind True --dynamic_ind_name zz1000_10_MST_0_spectral_ 
```

## Implemented Objective Functions 支持的目标函数

Currently this platform support the following four objective functions:

目前平台支持以下四种目标函数

### qp_method_1

objective: maximize return deducting risks

constraints: holding limits, style exposures, industry exposures, and turnover.

目标函数：最大化经风险调整后收益率

约束条件：持仓上下限、风格暴露约束、行业暴露约束、换手率约束

$$
\max_x \ \ R^T x - \lambda x^T \Sigma x \\
s.t. \ \ \forall i \ \ W_{low} \leq x_i \leq W_{high}, \ \ \sum_{i=1}^n x_i = 1 \ \ (\text{Holding Constraint}) \\
\forall k  \ \ S_{low} \leq (x^T - w_{bench}^T) X_{style_k} \leq S_{high}
\ \ (\text{Style Exposure}) \\
\forall k  \ \ I_{low} \leq (x^T - w_{bench}^T) X_{ind_k} \leq I_{high}
\ \ (\text{Industry Exposure}) \\
\sum_{i=1}^n |x - x_{t-1}| \leq TO_{limit} \ \ (\text{Turnover})
$$

### qp_method_2

objective: maximize return deducting risks and turnover.

constraints: holding limits, style exposures, and industry exposures.

目标函数：最大化经风险和交易成本调整后收益率

约束条件：持仓上下限、风格暴露约束、行业暴露约束

$$
\max_x \ \ R^T x - \lambda x^T \Sigma x - \theta \|x - x_{t-1}\|_1\\
s.t. \ \ \forall i \ \ W_{low} \leq x_i \leq W_{high}, \ \ \sum_{i=1}^n x_i = 1 \ \ (\text{Holding Constraint}) \\
\forall k  \ \ S_{low} \leq (x^T - w_{bench}^T) X_{style_k} \leq S_{high} \ \ (\text{Style Exposure}) \\
\forall k  \ \ I_{low} \leq (x^T - w_{bench}^T) X_{ind_k} \leq I_{high} \ \ (\text{Industry Exposrue}) \\
$$

More details to be discussed below (absolute value handling)

具体实现细节详见开发者日志（绝对值函数的处理）

### qp_method_3

objective: maximize return deducting risks, turnover, and style exposure

constraints: holding limits and industry exposure.

目标函数：最大化经风险,交易成本调整,风格约束后的收益率

约束条件：持仓上下限、行业暴露约束

$$
\max_x \ \ R^T x - \lambda x^T \Sigma x - \theta \|x - x_{t-1}\|_1 - \nu \left[ (x^T - w_{bench}^T) X_{style_k}X_{style_k}^T(x - w_{bench}) \right]\\
s.t. \ \ \forall i \ \ W_{low} \leq x_i \leq W_{high}, \ \ \sum_{i=1}^n x_i = 1 \ \ (\text{Holding Limit}) \\
\forall k  \ \ I_{low} \leq (x^T - w_{bench}^T) X_{ind_k} \leq I_{high} \ \ (\text{Industry Exposure}) \\
$$

### qp_method_4 (aka misalignment 模型错位)

objective: maximize return deducting risks and discarding portions of returns unaccounted returns by style factors to address model misalignment.

constraints: holding constraints, style exposures, industry exposures, and turnover limit.

目标函数：最大化经风险调整后收益率, 刨去未被风险因子解释的收益率

约束条件：持仓上下限、风格暴露约束、行业暴露约束、换手率约束

$$
\max_x \ \ R^T x - \lambda x^T \Sigma x - \xi (R_{ortho}^T x)^2\\
s.t. \ \ \forall i \ \ W_{low} \leq x_i \leq W_{high}, \ \ \sum_{i=1}^n x_i = 1 \ \ (\text{Holding Constraint}) \\
\forall k  \ \ S_{low} \leq (x^T - w_{bench}^T) X_{style_k} \leq S_{high} \ \ (\text{Style Exposure}) \\
\forall k  \ \ I_{low} \leq (x^T - w_{bench}^T) X_{ind_k} \leq I_{high} \ \ (\text{Industry Exposure}) \\
\sum_{i=1}^n |x - x_{t-1}| \leq TO_{limit} \ \ (\text{Turnover})
$$

More to be discussed in Factor Investing 7.3.1

详情请参考石川因子投资7.3.1错位的收益与风险模型

## For Developer 开发者日志

优化目标推导:
holding n stocks,

original objective function:
$$
\max_x \ \ R^T x - \lambda x^T \Sigma x - \theta |x-c|  \\
s.t. \ \ Gx \leq h, Ax = b
$$
which is equivalent to the following quadratic form (obj func 2)
$$
\begin{aligned}
&\min_{y =[y^+ , y^-]} \ \ \lambda y^T \left[
\begin{array}{c|c}
\Sigma & -\Sigma \\ \hline
-\Sigma & \Sigma
\end{array}\right] y + 2\lambda[\Sigma c, -\Sigma c] y + \theta [1,...,1]y - [R, -R]y \\
:= &\min_{y =[y^+ , y^-]} \ \ \frac{1}{2}  y^T 2\lambda\left[
\begin{array}{c|c}
\Sigma & -\Sigma \\ \hline
-\Sigma & \Sigma
\end{array}\right] y + [2\lambda\Sigma c + \theta_{n} - R, -2\lambda\Sigma c + \theta_{n} + R]y \\
\end{aligned} \\
s.t.  \ \ [[G, -G]; -I_{2n}] y \leq [h - Gc; 0_{2n}], \ \ [A, -A] y = b - Ac
$$
where
$$
y^+ = \max(x-c, 0) \\
y^- = \max(c-x, 0)
$$
so that
$$
y^+ + y^- = |x-c| \\
y^+ - y^- = x-c
$$
and
$$
x = y^+ - y^- +c
$$
