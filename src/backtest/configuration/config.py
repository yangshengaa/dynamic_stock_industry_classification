"""
backtest config 
"""

# load packages 
import os 
import sys
import getpass
import pandas as pd

# init dataserver
# from PqiDataSdk import *
sys.path.append("../..")
from data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

user = getpass.getuser()

# name output folder 
test_name = 'test'

'''
======= BackTest System Config ========
'''
max_processes = 8  # batch testing max processors to use
ds_max_processes = 12  # Dataserver读取数据的最大进程数

# TODO: initialize other dataset 
# myconnector = PqiDataSdk(user=user, size=ds_max_processes, pool_type="mp", log=False, offline=True)
myconnector = PqiDataSdkOffline()

'''
========= Path Config ==========
'''
# other factors read path
# TODO: 成分股df
index_member_stock_path = '../../../data/parsed/index_stock_weight'    # 成分股路径
output_path = '../res'                                                 # 回测结果存储路径
signal_df_output_path = '../signal_df'                                 # 模型持仓存放路径
risk_fac_data_path = '../../../data/features/risk_factor'              # barra因子读取路径
# TODO: 计算barra因子

# alpha factor read path 
factor_path = '../../../data/features/factor'


'''
============ BackTest Config =============
'''
# TODO: to English
start_date = '20210101'
end_date = '20210630'
adj_freq = 1  # 调仓周期
freq = "D"  # 调仓模式，W为按周，D为按日
group_num = 20 # 分组测试分组数量
head = 200 # 多空组分别选取的股票数
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
fixed_stock_pool = None  
if fix_stocks:
    # 固定票池名称，可选 "all", "300","500","800","1000","1200"
    if fixed_pool != "all":
        fixed_stock_pool = pd.read_table("../configuration/stock_list_{}.txt".format(fixed_pool)).columns[0].split(",")
    else:
        fixed_stock_pool = myconnector.get_ticker_list()

# 收益率基准设置 (是否用独立的指数作为基准)
use_independent_return_benchmark = False
return_benchmark_index = 'zz500'  


# riskplot对标收益序列 # !!! (如需使用riskplot，请先用因子生成平台生成风格因子值和因子收益率)
risk_plot_required = False  # 是否做riskplot分析
risk_plot_benchmark_index = '000852'  # riskplot归因对比序列
# 备注: 除中证1000外，其余都从2015年1月30号开始有权重；1000从15年5月29号开始有权重。请酌情设置回测区间

'''
=============== factor / signal names =============
'''

from configuration.factor_signal_test_list import *


'''
=============== others ======================
'''

# 其他中间变量
# TODO: other trade dates 
# trade_dates = myconnector.get_trade_dates(start_date=start_date, end_date=end_date)
trade_dates = myconnector.select_trade_dates(start_date, end_date)
