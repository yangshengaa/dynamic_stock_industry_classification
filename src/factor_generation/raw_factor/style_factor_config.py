"""
Barra Factor Config 
"""

# init data source 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
ds = PqiDataSdkOffline()

# index member stock weight path 
index_weight_path = 'data/parsed/index_stock_weight'
index_list = ['000016', '000300', '000905', '000852']  # 上证50，沪深300，中证500，中证1000
index_code_to_weight = {
    '000905': 'ZZ500_WGT.npy',   # 中证500
    '000852': 'ZZ1000_WGT.npy',  # 中证1000
    '000985': 'ZZall_WGT.npy',   # 中证全票
    '000016': 'SZ50_WGT.npy',    # 上证50
    '000300': 'HS300_WGT.npy'    # 沪深300
}

# barra factor save path 
saving_path =  'data/features/risk_factor' # barra风格因子存储路径

# specify dates
start_date = '20150101'
end_date = '20211231' 


















# ==================================
# ------ in construction ... -------
# ==================================

# 其他信息
# TODO: 更新财务数据字段
fund_data_list = ['CASH_RATIO', 'CR', 'EPS_BASIC_Q', 'FREE_CASHFLOW', 'NP_CUT_GROWTH', 'NP_CUT_Q',
                  'OP_ACT_CASH_GROWTH_B', 'OP_ACT_NET_CASH_Q', 'OP_INCO_GROWTH_TTM_B', 'OP_MARG_RATIO_TTM',
                  'OP_NP_RATIO_TTM',
                  'PER_SHARE_NET_CASH_TTM', 'QUICK_RATIO', 'RECE_ACC_TURN', 'ROA_TTM', 'ROE_TTM',
                  'TOT_ASS_TURN', 'TOT_OWNER_EQUITY', 'TOT_ASSETS', 'TOTAL_LIAB']

fund_data_list = [
    # leverage 
    'MARKET_LEVERAGE_TTM',       # MLEV
    'BOOK_LEVERAGE_TTM',         # BLEV
    'DEBT_TO_ASSET_RATIO_TTM',   # DTOA

    # quality 

    # values 
    'PB_RATIO_TTM',              # BTOP 

    # growth 
    'BASIC_EARNINGS_PER_SHARE',

    # earning yield
    'EP_RATIO_TTM', 
    'EBIT_TTM', 
    'CASH_EQUIVALENT_PER_SHARE_TTM',

    
]