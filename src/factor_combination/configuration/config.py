"""
Factor Combination Config 
"""

# load packages 
import os
import numpy as np
import pandas as pd

# load files 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
myconnector = PqiDataSdkOffline()

# ------ test range ------------
start_date = "20160101"  
end_date = "20211231"  
trade_dates = myconnector.select_trade_dates(start_date=start_date, end_date=end_date)

# set stock pool 
fixed_pool = "all"
index = 'zz1000'
# fix stock pool name "all", "300","500","800","1000","1200","1800","1800_out"
if fixed_pool != "all":
    stock_pool = pd.read_table("../configuration/stock_list_{}.txt".format(fixed_pool)).columns[0].split(",")
else:
    stock_pool = myconnector.get_ticker_list()


# -------- factor list ----------
# for past usage
# from src.factor_combination.configuration.factor_list import *  

# read all features 
features = os.listdir('data/features/factor')
features = [x[4:] for x in features]

# exclude some factors 
# features = [x for x in features if 'alpha' not in x.lower()]
# features = [x for x in features if 'signal' not in x]

# features = ['ATR_30', 'Alpha42', 'cokurt_40', 'mom_5']  # for testing purposes

# feature_dict = [features, features]
skip_features = None


# * -------- model selection -------- * 
# 'linear', 'xgb', 'Lgb', 'rf'
model_selection = 'linear'

# ————————traing config ————————
cost = 0.0015
whether_norm = True # true to map into (-1,1) 
whether_standardize = False # true to standardize
whether_importance = True # in xgb, true to output feature importance
nanmethod = "fill0" # nan processing: fill0 / ignore
neutralize = False # true to neutralize factors 
train_mode = "rolling"  # train mode: rolling / expanding / bisection
train_period = 240 # for rolling and expanding, the first training date length
test_period = 40  # for rolling and expanding, the first training date length
bisection = 1 # for bisection, the proportion to train 
valid_y = True # updown limit for y values (get feasible traidng stocks)
lag_date = 1 # lag 

# ———————— make y vague ——————— #
vague = True # true to make y vague around 0
vague_threshold = 0.001 # threshold to vague 
vague_rangerate = 2 # threshold for large y

# ——————— paths ———————
# factor path 
if neutralize:
    raise NotImplementedError('No Neutralized Factros Yet')
    factor_read_path = "/data/shared/low_fre_alpha/standardized_factors"
    features = ['neu_' + i for i in features]
else:
    # factor_read_path = "/data/shared/low_fre_alpha/tmpuse_yuhan"
    factor_read_path = "data/features/factor"

factor_save_path = "data/features/factor" # "/data/shared/low_fre_alpha/"
# output parameters 
model_params_path = 'out/model_param_record' #  f"../res/model_param_record/"
# index member stock weight path 
# eod_path =  # '/home/shared/Data/data/nas/eod_data/stock_eod_npy/eod/'
index_member_stock_path = 'data/parsed/index_stock_weight'
# log path
log_path = 'out/log' # f"../res/log/"
# 储存模型路径
model_path = 'out/model_res' #  f"../res/model_res/"
# 模型路径文档
model_record_path = 'out/log/model_record.csv' # f"../res/log/model_record.csv"
# 图片路径
fig_path = 'out/model_fig' # f"../res/fig/"

# ————————自定义收益率路径和文件名————————
diy_return = False # 是否采用固定票池
diy_return_path = f"/data/shared/low_fre_alpha/neutralized_y"
diy_return_name = "eod_op_rtn_6_mask_raw_style_normal"


# ————————设置XGB参数，手动修改————————
xgb_paras = {
    'num_round': 100,
    'max_depth': 5,
    'eta': 0.06,
    'subsample': 1,  # 随机采样训练样本 训练实例的子采样比
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1-0.2
    'reg_lambda': 1,  # 控制模型复杂度的权重值的L2正则化项参数
    # 最大增量步长，我们允许每个树的权重估计。
    'max_delta_step': 0,
    'colsample_bytree': 1,  # 生成树时进行的列采样
    'booster': 'gbtree',
    # 'tree_method': 'gpu_hist'
}

nest_xgb_paras_lst = [
{'n_estimators': 250,
 'max_depth': 5,
 'learning_rate': 0.06,
 'subsample': 1,  # 随机采样训练样本 训练实例的子采样比
 'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1-0.2
 'reg_lambda': 1,  # 控制模型复杂度的权重值的L2正则化项参数
 # 最大增量步长，我们允许每个树的权重估计。
 'max_delta_step': 0,
 'colsample_bytree': 1,  # 生成树时进行的列采样
 'booster': 'gbtree',
 'missing': np.nan,  # 自动处理nan
 'tree_method': 'gpu_hist'},

{'n_estimators': 250,
 'max_depth': 5,
 'learning_rate': 0.06,
 'subsample': 1,  # 随机采样训练样本 训练实例的子采样比
 'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1-0.2
 'reg_lambda': 1,  # 控制模型复杂度的权重值的L2正则化项参数
 # 最大增量步长，我们允许每个树的权重估计。
 'max_delta_step': 0,
 'colsample_bytree': 1,  # 生成树时进行的列采
 'booster': 'gbtree',
 'missing': np.nan,  # 自动处理nan
 'tree_method': 'gpu_hist'}
]
save_log = True
predict_type = 'Regressor'

# ———————— Nested model ———————— #
nested = False
nest_weight_method = ['SY3+TD6+None', 'LY3+TD6+None']
nest_type = ['Regressor', 'Regressor']

# ———————— sample weights mode ———————— #
weight_method = "EW"
# weight_method = "None+None+LID0" # EW / LYnum+TDnum, LY和TD处可以不写
decay_func = 'polynomial+1' # polynomial+num / exponential / linear

# ————————LGB Parameters ————————
lgb_paras = {'max_depth': 8, 'num_leaves': 50, 'min_child_samples' : 25,  'learning_rate': 0.05, 'n_estimators': 100, 'n_jobs' : 50}
lgb_rolling_feature_selection = False # 是否进行rolling的feature selection
lgb_num_feature = 100  # rolling的feature selection选择的因子数量
lgb_rolling_paras = False # 是否进行滚动调参

# ———————— RF parameters ————————
rf_paras = {
 'n_estimators': 200,
 'max_depth': 30,
 'min_samples_leaf': 25,
 'n_jobs': -1
}

# ———————— Linear Model Para Search ————————
alpha_range = np.logspace(-5, -2, 3, base = 10) # 参数选择范围
score_metric = 'IC'  # 模型评价指标：Regression: IC, R2等 ; Classification: f1, roc_auc等
self_defined_score = True  # 模型评价指标是否是自定义的：
                           # 若是(如IC)，则需在Evaluation.py中自定义指标并import到Model Collection.py
                           # 若否，则应直接输入sklearn里支持的scorer

# ————————lgb模型调参设置————————
lgb_paras_test = {
    'max_depth':  [8, 10, 12],
    'num_leaves': range(30, 60, 10),
    'learning_rate': [0.05, 0.1],
    #'n_estimators' :[50, 100],
    'min_child_samples': [15,20,25]
    #'subsample':[0.5, 0.75, 1],
    #'colsample_bytree':[0.5, 0.75, 1],
    #'reg_alpha':[0,1],
    #'reg_lambda':[0, 5],
} # 参数选择范围

# ————————遗传算法相关设置————————
# gene_fac_only = False
# gene_fac_flag = False
# gene_fac_dirs = ['/data/shared/low_fre_alpha/gene_tmp_factors/']

# ————————设置预测数据类型————————
return_type = "open_to_open"
# 设置收益率区间, e.g. -1至-4表示一天后开始的3日收益率
return_start_date = -1 # {return_start_date}天后开始的收益率
return_end_date = -2 # {return_end_date}天后结束的收益率



# # ————————————模型配置class（废弃）———————————— #
# import pandas as pd
#
# class config:
#
#     def __init__(self, init_dict = None):
#         # 设置训练开始和终止日期
#         self.start_date = "20200102"
#         self.end_date = "20201231"
#
#         # 设置票池
#         self.fix_stocks = True  # 是否采用固定票池
#         self.fixed_pool = "all"
#         self.index = "zz500"
#         if self.fix_stocks:
#             # 固定票池名称，可选 "all", "sz1200"
#             if self.fixed_pool == "all":
#                 self.sz1200 = pd.read_table("../configuration/stock_list_1200.txt").columns[0].split(",")
#         else:
#             self.index_pool = [self.index]  # 历史上当日中证500成分股，每日500个可选股票
#
#         # 设置因子组合的列表
#         self.features = ["mom_1", "mom_3", "range_rate_1", "range_rate_3"]
#         self.skip_features = None
#
#         # 路径设置
#         self.eod_path = "/home/shared/Data/data/nas/eod_data/stock_eod_npy/eod/"
#         self.factor_path = f"/home/yhzhou/data/low_fre_alpha/factors/"
#         self.save_path = f"/home/yhzhou/data/XgBoost/Xgb_low_freq_alpha_gen/res/"
#         self.log_path = f"/home/yhzhou/data/XgBoost/Xgb_low_freq_alpha_gen/res/log/"
#
#         # 设置配置参数
#         self.whether_norm = True
#         self.nanmethod = "fill0" # fill0 / ignore
#         self.ds_max_processes = 20
#         self.train_mode = "rolling" # rolling / expanding / dichotomy
#         self.train_period = 60
#         self.test_period = 20
#         self.dichotomy = 0.6
#
#         # 设置预测数据类型
#         self.return_type = "vwap_to_vwap"
#
#         # 外部自定义参数
#         if init_dict is not None:
#             for key in init_dict.keys():
#                 exec(f"self.{key} = init_dict['{key}']")
#
#


