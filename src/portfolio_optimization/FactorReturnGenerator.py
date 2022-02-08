"""
Compute Style Factor Returns
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
# # control numpy threads
# os.environ["MKL_NUM_THREADS"] = "10" 
# os.environ["NUMEXPR_NUM_THREADS"] = "10" 
# os.environ["OMP_NUM_THREADS"] = "10"
# os.environ["OPENBLAS_NUM_THREADS"] = "10"

# ignore warnings 
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')

# initialize dataserver 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

# load config
import src.portfolio_optimization.config as cfg

class FactorReturnGenerator(object):
    def __init__(self):
        # dates
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date

        # dataserver
        self.myconnector = PqiDataSdkOffline()

        # stock pool and dates
        self.all_stocks = self.myconnector.get_ticker_list()
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers=self.all_stocks, 
            start_date=self.start_date, 
            end_date=self.end_date
        )
        self.trade_dates = self.myconnector.select_trade_dates(
            start_date=self.start_date, 
            end_date=self.end_date
        )
        self.tickers = list(self.eod_data_dict["ClosePrice"].index)  
        self.date_list = list(self.eod_data_dict['ClosePrice'].columns)
        self.index_code_to_name = cfg.index_code_to_name

        # dynamic industry 
        self.use_dynamic_ind = cfg.use_dynamic_ind
        self.dynamic_ind_name = cfg.dynamic_ind_name
        
        # mode for return calculations 收益率回看模式
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

        # paths 路径
        self.class_factor_read_path = cfg.class_factor_path
        self.ret_save_path = cfg.ret_save_path

        # set up storing dicts 各类存储dict
        self.class_factor_dict_adj = {}  # 存放大类风格因子值 aggregated style factors 
        self.factor_return_df_dict = {}  # 存放大类风格因子值的收益 aggregated style factor returns
        self.idio_return_df_dict = {}    # 存放特质收益 idiosyncratic returns
     
    
    # =================== factors ========================

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

    # ======================= industry =========================

    def get_ind_data(self):
        """ retrieve static industry data """
        # TODO: add others (代码见low_fre_alpha_generator/process_raw/neutralize_factor)
        # TODO: add country factor? (CNE5)
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

    def get_dynamic_ind_data(self) -> pd.DataFrame:
        """ get dynamic industry data """
        ind_df = self.myconnector.read_ind_feature(self.dynamic_ind_name)
        ind_df_selected = ind_df[self.trade_dates]
        self.ind_df = ind_df_selected
    
    # ==================== index ========================

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
        read stock index weights and convert to mask. 1 if it is a member stock, nan otherwise.
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

    # ====================== calc ret =========================

    def calc_fac_ret(self, return_type):
        """ 计算风格因子收益率 (与riskplot平台不同，此处用的是WLS) """
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

        # static industry: copy once and replace columns in the loop 
        if not self.use_dynamic_ind:
            X = self.ind_df.copy()  # 外层copy一次，循环内自己换factor_name
        else:
            sample_cross_section = self.ind_df.iloc[:, 0]
            ind_labels = sorted(list(set(sample_cross_section.tolist())))
            map_dict = dict(zip(ind_labels, np.eye(len(ind_labels), dtype=int)))

        # TODO: add dynamic industry here 

        has_value_dates = []  # store dates that have values # * for local data only 
        for t in range(len(self.date_list) - shift_step):            
            # prepare data
            Y = ret_df.iloc[:, t + shift_step]
            #dt = fac.columns.tolist()[t+shift_step]

            # dynamic industry: extract days and one hot encode it 
            if self.use_dynamic_ind:
                X = pd.DataFrame(
                    self.ind_df.iloc[:, t].apply(lambda x: map_dict[x]).tolist(),
                    columns=ind_labels,
                    index=all_ticker
                )

            for class_name in self.class_factor_dict_adj.keys():
                X[class_name] = self.class_factor_dict_adj[class_name].iloc[:, t]
            today_size = self.eod_data_dict['FloatMarketValue'].iloc[:, t]   # TODO: 是否需要取log?
            all_df = pd.concat([Y, X, today_size], axis=1).dropna(axis = 0)
            ticker_lst = all_df.index
            data = all_df.values.T
            
            if len(data[0]) > 0:  # 目前必须当日要有数据，否则后面设置index的时候会报错
                has_value_dates.append(self.date_list[t])
                # ind risk 回归  
                # sm.WLS的weight是1/sqrt(w)            
                data_x, data_y, data_weight = data[1:-1].T, data[0], data[-1]
                model = sm.WLS(data_y, data_x, weights=data_weight)  # TODO: weights研究
                res = model.fit()
                f_1.append(res.params)  # 因子收益率

                residual = data_y - data_x.dot(res.params)
                f_2.append(residual)  # 特质收益率
                ticker_total_lst.append(ticker_lst) # 当日有值的ticker

        # find out nan dates
        nan_dates = list(set(self.date_list) - set(has_value_dates))

        # 拼接因子收益率
        if self.use_dynamic_ind:
            fac_name_list = [f'ind_{x}' for x in ind_labels] + list(self.class_factor_dict_adj.keys())
        else: 
            fac_name_list = list(self.ind_df.columns) + list(self.class_factor_dict_adj.keys())
        factor_return_df = pd.DataFrame(
            f_1, 
            columns=fac_name_list,
            # index=self.date_list[1:]  # * 针对目前的c2next_c实际上是last_c2c
            index=has_value_dates
        )
        # fill back nan
        nan_factor_return_df = pd.DataFrame(np.nan, columns=fac_name_list, index=nan_dates)
        factor_return_df = factor_return_df.append(nan_factor_return_df)
        factor_return_df.sort_index(inplace=True)  # sort on dates
        self.factor_return_df_dict[return_type] = factor_return_df
        
        # 拼接特质收益率  # * 针对目前的c2next_c实际上是last_c2c
        idio_return_df = pd.concat(
            # [pd.Series(each_f_2, index=each_ticker, name=each_date) for each_f_2, each_ticker, each_date in zip(f_2, ticker_total_lst, self.date_list[1:])], 
            [pd.Series(each_f_2, index=each_ticker, name=each_date) for each_f_2, each_ticker, each_date in zip(f_2, ticker_total_lst, has_value_dates)],
            axis=1
        ).T  # columns为ticker，index为日期
        idio_return_df[list(set(self.tickers) - set(idio_return_df.columns))] = np.nan  # 给剩下的都附上nan
        idio_return_df = idio_return_df.sort_index(axis = 1)
        # fill back nan
        nan_idio_return_df = pd.DataFrame(np.nan, columns=self.tickers, index=nan_dates)
        idio_return_df = idio_return_df.append(nan_idio_return_df)
        idio_return_df.sort_index(inplace=True)  # sort on dates
        self.idio_return_df_dict[return_type] = idio_return_df


    def save_ret(self):
        '''
        把收益率储存到本地
        '''
        for return_type in self.return_type_list:
            # calc path 
            ret_save_path = os.path.join(self.ret_save_path, self.dynamic_ind_name)
            # make new dir 
            if not os.path.isdir(ret_save_path):
                os.mkdir(ret_save_path)

            # save
            self.factor_return_df_dict[return_type].reset_index().to_feather(
                '{}/Factor_return_{}_'.format(ret_save_path, return_type) # , 
                # float_format='%.8f'
            )
            self.idio_return_df_dict[return_type].reset_index().to_feather(
                '{}/Idio_return_{}_'.format(ret_save_path, return_type) # , 
                # float_format='%.8f'
            )
        
    
    def start_cal_return_process(self):
        """ 运行主函数 """
        print("Reading Style Factors 正在读取风格因子")
        t0 = time.time()
        self.load_factor_data()
        if self.use_dynamic_ind: # dynamic 
            self.get_dynamic_ind_data()
        else: # static 
            self.get_ind_data()
        print("Style Factor Reading Takes 读取风格因子耗时", time.time() - t0)
        print("Calculating Style Factor Returns and Idio Returns 正在计算风格行业因子收益率和特质收益率")
        t0 = time.time()
        for ret_type in self.return_type_list:
            self.calc_fac_ret(ret_type)
            
        print("Return Calculation Takes 计算风格行业因子收益率和特质收益率耗时", time.time() - t0)
        
        # 数据结果落地
        print("Storing 正在储存数据")
        t0 = time.time()
        self.save_ret()
        print("Storing takes 储存数据耗时", time.time() - t0)
