"""
portfolio optimization config. In sequence, we have 

- FactorReturnGenerator.py 
- CovMatrixEstimator.py 
- WeightOptimizer.py
- PlotRisk.py
"""


# —————— data path ——————
class_factor_path = "data/features/risk_factor/class_factors"  # aggregated style factor path
ret_save_path = 'out/return'  # factor return store path
cov_save_path = 'out/cov'  # covariance matrix store path f"../res/cov/"
input_signal_path = 'out/signal_df'   # input signal df
output_signal_path = 'out/signal_df'    # output signal df
fig_save_path = 'out/risk_fig/'
ml_factor_path = 'data/features/ml_factors' # predicted return path
index_member_stock_path = 'data/parsed/index_stock_weight'

start_date = '20170701'
end_date = '20211231'

return_type_list =  ['c2next_c'] # ['o2next_o', 'c2next_o', 'o2c', 'c2next_c']

# aggregated 
class_name = [
    'WeightedMomentum', 
    'Volatility', 
    'Turnover', 
    'Size', 
    'NLSize', 
    'Momentum',
    'Reversal',
    'IndustryMomentum', 
    'err_std', 
    'betas', 
    'alpha'
] 


# —————— cov estimation parameters —————
h = 240   # sample 240

tau = 90   # EWMA half life 90
lam = 0.5 ** (1/tau)   # EWMA weight


N_W = True  # 是否进行Newey-West自相关调整
pred_period = 1  # Newey-West自相关调整预测风险时长
D = 4   # Newey-West自相关调整总滞后时长  # * 不同因子收益率的时序变化非常大, 可以直接用 4 * (240 / 100) ** (2 / 9), 向下取整为4

eigen_adj = True   # 是否进行因子协方差矩阵特征值调整
alpha = 1.2  # 模拟风险偏差调整系数（取值略大于1,研报经验值1.2）

struc_adj = True # 是否进行结构化调整
h_struc = 240  # 结构化调整 时间周期
min_o = 50  # 最小样本数量
E0 = 1.05  # 残差项偏误常数

bs_adj = True   # 是否进行特质协方差矩阵贝叶斯调整
q = 1  # 贝叶斯经验压缩系数
bs_group_num = 10 # 根据市值将股票分为几组

vol_adj = True   # 是否进行波动率偏误调整
h_vol = 60   # 波动率偏误调整 时间周期 240
tau_vol = 20  # 波动率偏误调整 半衰期 40
lam_vol = 0.5 ** (1/tau_vol)   # 波动率偏误调整 EWMA权重




# —————— portfolio optimization ——————
# TODO: use a different signal and ml factor
# input_signal_df_name = 'xgb_agg.csv'  # 输入信号名字，信号格式为0/1 
input_signal_df_name = 'lgb_161227_211231_240_40_202201240953_Regressor_zz1000_fmv_100_1/long_0'
# input_signal_df_name = 'lr_bm_161227_211231_240_40_202201241841_Regressor_linear_0_rolling_zz1000_fmv_100_1/long_0'
obj_func = 'ret_var' # 目标函数：min_var (最小方差)/ ret_var (均值-方差优化)

weight_low = 0.001   # 权重下限
weight_high = 0.05  # 权重上限

# ml_factor_name = 'xgb_161227_201231_240_40_202109090912_Regressor'
ml_factor_name = 'lgb_161227_211231_240_40_202201240953_Regressor'
# ml_factor_name = 'lr_bm_161227_211231_240_40_202201241841_Regressor_linear_0_rolling'

benchmark_index = 'zz1000'  # 风格行业中性约束的基准指数

style_neutralize = True  # 风格中性约束
style_low_limit = -0.4  # 风格中性因子敞口  # TODO: 放0.5以内
style_high_limit = 0.4

ind_neutralize = True  # 行业中性约束
ind_low_limit = -0.1 # 行业中性因子敞口
ind_high_limit = 0.1

turnover_constraint = True  # 换手率约束
turnover_limit = 0.4

penalty_lambda = 10  # 风险厌恶系数   # TODO: 不同的风险系数
penalty_theta = 0.1  # 换仓惩罚  # 仅在method 2 中使用 # 如果用的是0.015/2， 则该项刚好为实盘cost
penalty_nu = 100     # 风格暴露惩罚  # 仅在method 3 中使用
penalty_xi = 1       # 模型错位风险惩罚  # 仅在method 4 中使用

qp_method = 1  # 目标函数和限制方法

adj_method = '_NW1_4_Eigen_struc_240_bs_vol60_20'  # 读取对应调整方法的协方差矩阵 '_NW3', '_Eigen_bs_vol', 目前均使用'_NW1_4_Eigen_bs_vol60_20'



# —————— plotting ——————
risk_cal_day = 10  # 计算日波动率的天数
plot_pred_period = 1  # 计算偏差统计量时的风险周期（eg：20 代表月频风险）,应与使用的NW pred_period对应

self_input_weight = False  # 若为True,则读取输入csv的持仓和权重；若为False，则使用以下设置的指数持仓和权重
input_weight_df_name = 'xgb_test.csv' # 输入持仓权重文件名
index = 'zz1000' # 动态持仓的指数
holding_weight = 'FMV' # 持仓权重 'EW':等权 'FMV':流通市值加权

benchmark_weight = 'EW' # 比较基准持仓权重

test_name = 'struc_adjust_test' # 输出文件夹名字


# —————— assert ——————
assert penalty_lambda > 1e-6 


# ------ others ------

# industry code to name 
index_code_to_name = {
    '801010': '农林牧渔',
    '801020': '采掘',
    '801030': '化工',
    '801040': '钢铁',
    '801050': '有色金属',
    '801080': '电子',
    '801110': '家用电器',
    '801120': '食品饮料',
    '801130': '纺织服装',
    '801140': '轻工制造',
    '801150': '医药生物',
    '801160': '公用事业',
    '801170': '交通运输',
    '801180': '房地产',
    '801200': '商业贸易',
    '801210': '休闲服务',
    '801230': '综合',
    '801710': '建筑材料',
    '801720': '建筑装饰',
    '801730': '电气设备',
    '801740': '国防军工',
    '801750': '计算机',
    '801760': '传媒',
    '801770': '通信',
    '801780': '银行',
    '801790': '非银金融',
    '801880': '汽车',
    '801890': '机械设备'
}
