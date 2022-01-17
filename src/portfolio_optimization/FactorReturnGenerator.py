"""
计算风格因子收益率
"""

# load packages 
import os
import time
import datetime
import logging
import warnings
import traceback
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# # 控制numpy进程数
# os.environ["MKL_NUM_THREADS"] = "10" 
# os.environ["NUMEXPR_NUM_THREADS"] = "10" 
# os.environ["OMP_NUM_THREADS"] = "10"
# os.environ["OPENBLAS_NUM_THREADS"] = "10"

# 绘图全局设置
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 10
# ignore warnings 
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')

# initialize dataserver 
# from PqiDataSdk import *
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
# from getpass import getuser

# load file
import src.portfolio_optimization.config as cfg


class FactorReturnGenerator(object):
    def __init__(self):
        # 基本常量
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        # self.myconnector = PqiDataSdk(
        #     user=getuser(), 
        #     size=10, 
        #     pool_type="mp", 
        #     log=False, 
        #     offline=True, 
        #     str_map=False
        # )
        self.myconnector = PqiDataSdkOffline()
        all_stocks = self.myconnector.get_ticker_list()
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers=all_stocks, 
            start_date=self.start_date, 
            end_date=self.end_date
        )
        self.trade_dates = self.myconnector.select_trade_dates(
            start_date=self.start_date, 
            end_date=self.end_date
        )
        self.tickers = list(self.eod_data_dict["ClosePrice"].index)  
        self.date_list = list(self.eod_data_dict['ClosePrice'].columns)
        
        # 收益率回看模式
        self.ret_df_dict = {}    
        self.return_type_list = cfg.return_type_list
        self.Open = self.eod_data_dict["OpenPrice"] * self.eod_data_dict["AdjFactor"]
        self.Close = self.eod_data_dict["ClosePrice"] * self.eod_data_dict["AdjFactor"]

        if 'o2next_o' in self.return_type_list:
            self.ret_df_dict['o2next_o'] = self.Open.shift(-1, axis=1) / self.Open - 1
        if 'c2next_o' in self.return_type_list:
            self.ret_df_dict['c2next_o'] = self.Open.shift(-1, axis=1) / self.Close - 1
        if 'o2c' in self.return_type_list:
            self.ret_df_dict['o2c'] = self.Close / self.Open - 1
        if 'c2next_c' in self.return_type_list:
            self.ret_df_dict['c2next_c'] = self.Close / self.Close.shift(1, axis=1) - 1   # * 目前用这个，实际上是last_ctoc
            #self.ret_df_dict['c2next_c'] = self.Close.shift(-1, axis = 1) / self.Close - 1

        # 不同收益率与因子对齐需要shift的天数 
        # self.shift_step_dict = {'o2next_o': 1, 'c2next_o': 1, 'o2c': 0}
        # self.ret_df = self.ret_df_dict['o2next_o'].shift(1, axis=1)
        self.ret_df = self.ret_df_dict[self.return_type_list[0]]
        # self.saving_path = '/home/wlchen/data/'
        self.ind_df = pd.DataFrame()  # 行业0-1矩阵
        self.class_name = cfg.class_name

        # 路径
        self.class_factor_read_path = cfg.class_factor_path
        self.ret_save_path = cfg.ret_save_path

        # 各类存储dict
        self.class_factor_dict_adj = {}  # 存放大类风格因子值
        self.factor_return_df_dict = {}  # 存放大类风格因子值的收益
        self.idio_return_df_dict = {}    # 存放特质收益
     

    def read_factor_data(self, feature_name, tickers, date_list):
        """ 
        read a single aggregated style factor 
        :param feature_name: name for the style factor
        """
        # feature_name = "eod_" + feature_name
        # factor = self.myconnector.get_eod_feature(
        #     fields=feature_name, 
        #     where=self.class_factor_read_path, 
        #     tickers=tickers, 
        #     dates=date_list
        # )
        # return factor[feature_name].to_dataframe()
        factor_df = self.myconnector.read_eod_feature(
            feature_name, des='risk_factor/class_factors', dates=date_list
        )
        return factor_df


    def load_factor_data(self):
        """
        load all style factors
        """
        for name in self.class_name:
            self.class_factor_dict_adj[name] = self.read_factor_data(name, self.tickers , self.date_list)
    

    # def get_ind_date(self):
    #     # 读取行业数据
    #     # TODO: 滚动变化的行业？
    #     # TODO: add others (代码见low_fre_alpha_generator/process_raw/neutralize_factor)
    #     # TODO: add country factor? (CNE5)
    #     ind_members = self.myconnector.get_sw_members(level=[1])[["industry_name", "con_code"]]
    #     self.ind_df = pd.DataFrame(index=self.ret_df.index, columns=sorted(list(set(ind_members["industry_name"]))))
    #     for i in range(len(ind_members.index)):
    #         label = list(ind_members.iloc[i])
    #         self.ind_df.loc[label[1], label[0]] = 1
    #     self.ind_df = self.ind_df.fillna(0)
    #     # ind_name_list = ind_df.columns
    #     self.ind_df.columns = ["ind_" + str(i + 1) for i in range(len(self.ind_df.columns))]

    def get_ind_date(self):
        # 读取行业数据
        # TODO: 滚动变化的行业？
        # TODO: add others (代码见low_fre_alpha_generator/process_raw/neutralize_factor)
        # TODO: add country factor? (CNE5)
        # ind_members = self.myconnector.get_sw_members(level=[1])[["industry_name", "con_code"]]
        # self.ind_df = pd.DataFrame(index=self.tickers, columns=sorted(list(set(ind_members["industry_name"]))))
        # for i in range(len(ind_members.index)):
        #     label = list(ind_members.iloc[i])
        #     self.ind_df.loc[label[1], label[0]] = 1
        # self.ind_df = self.ind_df.fillna(0)
        # # ind_name_list = ind_df.columns
        # self.ind_df.columns = ["ind_" + str(i + 1) for i in range(len(self.ind_df.columns))]
        
        ind_members = self.myconnector.get_sw_members().drop_duplicates(subset=['con_code'])[["index_code", "con_code"]]
        self.ind_df = pd.DataFrame(
            index=self.all_stocks, columns=list(set(ind_members["index_code"]
        )))
        self.ind_df = self.ind_df[self.index_code_to_name.keys()]
        for i in range(len(ind_members.index)):
            label = list(ind_members.iloc[i])
            self.ind_df.loc[label[1], label[0]] = 1
        self.ind_df = self.ind_df.fillna(0)
        self.ind_df.columns = ["ind_" + str(i + 1) for i in range(len(self.ind_df.columns))]


    # 动态票池
    # def get_stock_weight(self, index):
    #     """
    #     获取该指数在测试区间内的成分股权重
    #     :param index:
    #     :return: 一个大dataframe, index为股票名，column是日期，value是个股权重
    #     """
    #     # 读取对应的npy
    #     index_to_weight = {
    #         'zz500': 'ZZ500_WGT.npy',
    #         'zz800': 'ZZ800_WGT.npy',
    #         'zz1000': 'ZZ1000_WGT.npy',
    #         'zzall': 'ZZall_WGT.npy',
    #         'sz50': 'SZ50_WGT.npy',
    #         'hs300': 'HS300_WGT.npy'
    #     }
    #     eod_path = '/home/shared/Data/data/nas/eod_data/stock_eod_npy/eod'
    #     index_file = index_to_weight[index]

    #     # 读入并转成dataframe形式
    #     index_data = np.load(os.path.join(eod_path, index_file))
    #     tickers = np.load(os.path.join(eod_path, 'ticker_names.npy'))
    #     dates = np.load(os.path.join(eod_path, 'dates.npy'))
    #     df = pd.DataFrame(index_data, columns=dates, index=tickers)
        
    #     # 与设置的时间段对齐
    #     df_selected = df.loc[:, self.trade_dates]
    #     return df_selected 
    
    def get_stock_weight(self, index: str) -> pd.DataFrame:
        """
        get stock weights of an given index within a timeframe 
        :param index
        :return: a dataframe, stock codes as index, dates as columns, and stock weights as values
        """
        # convert name to code 
        index_to_weight = {
            'zz500': '000905',
            'zz1000': '000852',
            'zzall': '000985',
            'sz50': '000016',
            'hs300': '000300'
        }
        # specify path 
        index_path = os.path.join(cfg.index_member_stock_path, index_to_weight[index])
        # retrieve 
        index_data = pd.read_feather(index_path).set_index("index")
        df_selected = index_data.loc[:, self.trade_dates]
        return df_selected

    
    def get_index_mask(self, index_list):
        """
        读取指数成分股权重，转换成指数成分股mask: 当天某股票为该指数成分股为1，否则为nan
        :param index_list: 一个装有指数的列表，支持组合指数
        :return 指数mask
        """
        agg_index_mask = False
        # 读入和拼接多个index
        for index in index_list:
            index_weight = self.get_stock_weight(index)
            index_mask = index_weight.notna()
            agg_index_mask = agg_index_mask | index_mask
        # 将dataframe转换为1和nan的矩阵
        agg_index_mask = agg_index_mask.astype(int) / agg_index_mask.astype(int)
        return agg_index_mask        


    def calc_fac_ret(self, return_type):
        """
        计算风格因子收益率 (与riskplot平台不同，此处用的是WLS)
        """
        # 获取收益率
        ret_df = self.ret_df_dict[return_type]
        
        # 动态票池mask
        # temp_index_mask = self.get_index_mask(index_list = ['zz500', 'zz1000'])
        # temp_index_mask = temp_index_mask + (self.eod_data_dict['ClosePrice'] - self.eod_data_dict['ClosePrice'])
        # ret_df = ret_df * temp_index_mask
        # print(temp_index_mask.shape[0] - temp_index_mask.isnull().sum())
        # print(ret_df.shape[0] - ret_df.isnull().sum())

        # 分日回归
        shift_step = 0
        all_ticker = ret_df.index
        #sample_data = pd.DataFrame([np.nan for _ in range(len(all_ticker))], columns=['sample'], index=all_ticker)
        f_1 = []  # 存放因子收益率
        f_2 = []  # 存放特质收益率
        ticker_total_lst = []  # 存放非nan的ticker
        X = self.ind_df.copy()  # 外层copy一次，循环内自己换factor_name
        for t in range(len(self.date_list) - shift_step):
            Y = ret_df.iloc[:, t + shift_step]
            #dt = fac.columns.tolist()[t+shift_step]
            for class_name in self.class_factor_dict_adj.keys():
                X[class_name] = self.class_factor_dict_adj[class_name].iloc[:, t]
            today_size = self.eod_data_dict['FloatMarketValue'].iloc[:, t]   # TODO: 是否需要取log?
            all_df = pd.concat([Y, X, today_size], axis=1).dropna(axis = 0)
            ticker_lst = all_df.index
            data = all_df.values.T
            
            if len(data[0]) > 0:  # 目前必须当日要有数据，否则后面设置index的时候会报错
                # ind risk 回归  
                # sm.WLS的weight是1/sqrt(w)            
                data_x, data_y, data_weight = data[1:-1].T, data[0], data[-1]
                model = sm.WLS(data_y, data_x, weights=data_weight)  # TODO: weights研究
                res = model.fit()
                f_1.append(res.params)  # 因子收益率

                residual = data_y - data_x.dot(res.params)
                f_2.append(residual)  # 特质收益率
                ticker_total_lst.append(ticker_lst) # 当日有值的ticker
        
        # 拼接因子收益率
        fac_name_list = list(self.ind_df.columns) + list(self.class_factor_dict_adj.keys())
        factor_return_df = pd.DataFrame(
            f_1, 
            columns=fac_name_list,
            index=self.date_list[1:]  # * 针对目前的c2next_c实际上是last_c2c
        )
        self.factor_return_df_dict[return_type] = factor_return_df
        
        # 拼接特质收益率
        # * 针对目前的c2next_c实际上是last_c2c
        idio_return_df = pd.concat(
            [pd.Series(each_f_2, index=each_ticker, name=each_date) for each_f_2, each_ticker, each_date in zip(f_2, ticker_total_lst, self.date_list[1:])], 
            axis=1
        ).T  # columns为ticker，index为日期
        idio_return_df[list(set(self.tickers) - set(idio_return_df.columns))] = np.nan  # 给剩下的都附上nan
        idio_return_df = idio_return_df.sort_index(axis = 1)
        self.idio_return_df_dict[return_type] = idio_return_df


    def save_ret(self):
        '''
        把收益率储存到本地
        '''
        for return_type in self.return_type_list:
            if not os.path.isdir(self.ret_save_path):
                os.mkdir(self.ret_save_path)
            self.factor_return_df_dict[return_type].to_csv('{}/Factor_return_{}_.csv'.format(self.ret_save_path, return_type))
            self.idio_return_df_dict[return_type].to_csv('{}/Idio_return_{}_.csv'.format(self.ret_save_path, return_type))
        
    
    def start_cal_return_process(self):
        """
        运行主函数
        """
        print("正在读取风格因子")
        t0 = time.time()
        self.load_factor_data()
        self.get_ind_date()
        print("读取风格因子耗时", time.time() - t0)
        print("正在计算风格行业因子收益率和特质收益率")
        t0 = time.time()
        for ret_type in self.return_type_list:
            self.calc_fac_ret(ret_type)
            
        print("计算风格行业因子收益率和特质收益率耗时", time.time() - t0)
        
        # 数据结果落地
        print("正在储存数据")
        t0 = time.time()
        self.save_ret()
        print("储存数据耗时", time.time() - t0)


# if __name__ == '__main__':
#     loading_process = FactorReturnGenerator()
#     loading_process.start_cal_return_process()
