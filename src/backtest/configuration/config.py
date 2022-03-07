"""
backtest config 
"""

# load packages 
import os 
import sys
import getpass
import pandas as pd

# init dataserver
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

user = getpass.getuser()

# name output folder 
test_name = 'short_experiment'

'''
======= BackTest System Config ========
'''
max_processes = 8  # batch testing max processors to use
ds_max_processes = 12  # Dataserver读取数据的最大进程数

# init data server 
myconnector = PqiDataSdkOffline()

'''
========= Path Config ==========
'''
# other factors read path
index_member_stock_path = 'data/parsed/index_stock_weight'    # member stock path
output_path = 'out/res'                                       # backtest record path 
signal_df_output_path = 'out/signal_df'                       # factor/signal holding path
risk_fac_data_path = 'data/features/risk_factor'              # barra risk factor path

# alpha factor read path 
factor_path = 'data/features/factor'  # for alpha factors

# signal factor read path 
factor_path = 'out/signal_df'           # for holding signals test
factor_path += '/lgb_161227_211231_240_40_202201280254_Regressor_zz1000_fmv_100_1'
# factor_path += '/lr_bm_161227_211231_240_40_202201241841_Regressor_linear_0_rolling_zz1000_fmv_100_1'

# # ml factor path 
# factor_path = 'data/features/ml_factor' 

'''
============ BackTest Config =============
'''
# TODO: to English
start_date = '20170701'
end_date = '20211231'
adj_freq = 1  # 调仓周期
freq = "D"  # 调仓模式，W为按周，D为按日
group_num = 10 # 分组测试分组数量
head = 100 # 多空组分别选取的股票数
cost = 0.0015 # 双边手续费
return_type = 'open_to_open' # 收益的回看模式   # 建议都用oto，因为自选指数暂时只有oto
ic_decay = 20 # IC Decay. 想看长周期可选100，短周期可选10
lag_decay = "lag"
lead_lag_list = [[-2,-1,0],[0,1,2,3,4]] # 因子lag不同期数的绩效
decay_list = [1,3] # 多空测试不同持仓时间的绩效.
max_lookback = 60  # 最大回看长度(为了剔除新股）
max_lookforward = 3 # 最大向前看长度（为了取收益率）

'''
=========== 票池设置 ==============
需要选择 交易票池 和 基准票池 两个票池
交易票池: 自选（静态）或动态票池（追踪指数历史成分股）
基准票池: 独立的指数收益率序列 或 动态票池对应指数的收益率序列（支持混合不同指数）
'''
# - weight == True时
#   - 如果只有一个index, 直接读取指数的oto收益率序列
#   - 如果多与一个index, 用fmv撮合指数收益序列
# - weight == False时, 不管有多少个index都是等权计算

# 交易票池设置
# 动态
# TODO: 添加票池文件
weight_index_by_fmv = True     # 混合票池，即index_list的length大于1时，是否用流通市值加权；否则等权
index_list = ['zz1000']    # 支持的票池如下，同时支持组合票池; 如果单票池就为一个length==1的list
index_dict = {           # 可选票池如下
    "sz50": "000016",    # 上证50   (这个票池没什么意义)
    "hs300": "000300",   # 沪深300 
    "zz500": "000905",   # 中证500  
    "zz1000": "000852",  # 中证1000     
    "zzall": "000985",   # 中证全票 (和自建全票不同，中证全票选择了全A并剔除ST及上市未满3个月的股票，权重为quantized的流通市值)
}
fix_stocks = False  # 如果fix_stocks为 False, 交易的票池与收益率基准票池相同（动态）

# TODO: 添加静态
# 静态（如果fix_stocks = True)
fixed_pool = '1000'
fixed_stock_pool = myconnector.get_ticker_list()
if fix_stocks:
    # 固定票池名称，可选 "all", "300","500","800","1000","1200"
    if fixed_pool != "all":
        fixed_stock_pool = pd.read_table("../configuration/stock_list_{}.txt".format(fixed_pool)).columns[0].split(",")
    else:
        fixed_stock_pool = myconnector.get_ticker_list()

# 收益率基准设置 (是否用独立的指数作为基准)
use_independent_return_benchmark = False
return_benchmark_index = 'zz500'  

# TODO: edit risk plots
# riskplot对标收益序列 # !!! (如需使用riskplot，请先用因子生成平台生成风格因子值和因子收益率)
risk_plot_required = False  # 是否做riskplot分析
risk_plot_benchmark_index = '000852'  # riskplot归因对比序列
# 备注: 除中证1000外，其余都从2015年1月30号开始有权重；1000从15年5月29号开始有权重。请酌情设置回测区间

'''
=============== factor / signal names =============
'''
# for signal in particular, if read from feather
read_from_feather = True 

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from factor_signal_test_list import *


'''
=============== others ======================
'''

trade_dates = myconnector.select_trade_dates(start_date, end_date)
