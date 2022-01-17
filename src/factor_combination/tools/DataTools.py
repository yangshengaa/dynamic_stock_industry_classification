"""
Datatools for factor combination module
"""

# load packages
import os
import sys
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from scipy.stats import rankdata
from sklearn import decomposition

# load files 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
from src.factor_combination.configuration import config as cfg
# from PqiDataSdk import *


class DataTools:
    # ———————— Utils ———————— #
    def __init__(
            self,
            predict_type: str='Regressor',
            features: List[str]=cfg.features, 
            stock_pool: List[str]=cfg.stock_pool
        ):
        # dates 
        self.index = cfg.index
        self.start_date = cfg.start_date # 总period开始时间
        self.end_date = cfg.end_date # 总period结束时间

        # paths 
        self.features = features # 涉及的因子列表
        self.factor_path = cfg.factor_read_path # 储存因子数据的路径
        self.return_type = cfg.return_type # y_label采用的收益率类别 (vwap_to_vwap, close_to_close, ...)
        self.whether_norm = cfg.whether_norm
        self.whether_standardize = cfg.whether_standardize
        self.nanmethod = cfg.nanmethod
        self.neutralize = cfg.neutralize
        self.skip_features = cfg.skip_features
        self.type = predict_type
        self.cost = cfg.cost
        # self.gene_features = []
        self.valid_y = cfg.valid_y
        self.return_start_date = cfg.return_start_date
        self.return_end_date = cfg.return_end_date
        self.vague = cfg.vague

        self.diy_return = cfg.diy_return
        self.myconnector = cfg.myconnector
        self.stock_pool = stock_pool
        
        # others 
        self.date_list = self.get_trade_days()

        # stock eod data 
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers = self.stock_pool,
            start_date = self.start_date, 
            end_date = self.end_date
        )
        self.return_data = self.get_return_data()

        # get factors
        self.factor_dict = self.get_eod_feature(tickers=self.stock_pool, date_list=self.date_list)
        self.ind_df = self.get_sw_ind_df()
        for k in self.eod_data_dict.keys():
            self.eod_data_dict[k].columns = [str(x) for x in self.eod_data_dict[k].columns]
        self.eod_data_dict["ind_df"] = self.ind_df

        self.lag_date = cfg.lag_date
        df_columns = []
        # if not cfg.gene_fac_only:
        for feature in self.features:
            for date in range(self.lag_date):
                df_columns.append(feature + "_" + str(date))
        # else:
        #     try:
        #         assert cfg.gene_fac_flag
        #     except AssertionError:
        #         print("Gene flag Conflict")
            # genetic algorithms 

        # if cfg.gene_fac_flag:
        #     print("Gennetic Factors are added and fac path is {}".format(cfg.gene_fac_dirs))
        #     for feature in self.gene_features:
        #         for date in range(self.lag_date):
        #             df_columns.append(feature + "_" + str(date))
        self.features = df_columns

    '''
    ———————————————————————— Reading Data ————————————————————————————
    '''

    # TODO: edited here 
    # @staticmethod
    # def get_stock_weight(index):
    #     """
    #     read index member stock weight 

    #     :param index:
    #     :return: a dataframe, with index for stock codes, column for dates，values are weights
    #     """
    #     eod_path = cfg.eod_path
    #     index_file = None
    #     if index == 'zz500':
    #         index_file = 'ZZ500_WGT.npy'
    #     if index == 'zz1000':
    #         index_file = 'ZZ1000_WGT.npy'
    #     if index == 'hs300':
    #         index_file = 'HS300_WGT.npy'

    #     index_data = np.load(os.path.join(eod_path, index_file))
    #     tickers = np.load(os.path.join(eod_path, 'ticker_names.npy'))
    #     dates = np.load(os.path.join(eod_path, 'dates.npy'))
    #     df = pd.DataFrame(index_data, columns=dates, index=tickers)
    #     return df.dropna(how='all')
    
    @staticmethod
    def get_stock_weight(index: str) -> pd.DataFrame:
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
        df_selected = index_data.loc[:, cfg.trade_dates]
        return df_selected


    def get_trade_days(self):
        """
        get trade dates between start_date and end_date
        """
        trade_dates = self.myconnector.select_trade_dates(
            start_date=self.start_date, 
            end_date=self.end_date
        )
        return trade_dates

    def get_eod_feature(self, tickers: List[str], date_list: List[str]) -> Dict[str, pd.DataFrame]:
        """
        get factor df within the time frame 
        """
        # ————————获取对应股票池和交易日的因子值———————— #
        # 以dict形式储存, return: {fac_name: pd.DataFrame}
        # path = self.factor_path
        # feature_name_list = ["eod_" + feature_name for feature_name in self.features]
        # factors = self.myconnector.get_eod_feature(fields = feature_name_list,
        #                                                where = path,
        #                                                tickers = tickers,
        #                                                dates = date_list)
        # factor_dict = {}

        # for factor in feature_name_list:
        #     if not self.diy_return:
        #         factor_dict[factor[4:]] = factors[factor].to_dataframe()
        #     else:
        #         factor_dict[factor[4:]] = self.y_mask * factors[factor].to_dataframe()
        #     factor_dict[factor[4:]][np.isinf(factor_dict[factor[4:]])] = np.nan

        # retrieve all factors
        factor_dict = {}
        for factor_name in self.features:
            factor_df = self.myconnector.read_eod_feature(
                factor_name, dates=date_list
            )
            factor_df = factor_df + factor_df * 0  # remove inf 
            factor_dict[factor_name] = factor_df

        return factor_dict

    def get_return_data(self) -> pd.DataFrame:
        """ obtain different returns """
        if not self.diy_return:
            self.price_df = pd.DataFrame()
            if self.return_type == 'open_to_open':
                self.price_df = self.eod_data_dict['OpenPrice'] * self.eod_data_dict['AdjFactor']
                self.eod_data_dict["OpenToOpenReturn"] = self.price_df.shift(self.return_end_date, axis=1) / self.price_df.shift(self.return_start_date, axis=1) - 1
                self.eod_data_dict["OpenToOpenReturn"] = self.eod_data_dict["OpenToOpenReturn"] - self.eod_data_dict[
                    "OpenToOpenReturn"].mean()
                if not self.valid_y:
                    return self.eod_data_dict["OpenToOpenReturn"]
            elif self.return_type == 'close_to_close':
                self.price_df = self.eod_data_dict['ClosePrice'] * self.eod_data_dict['AdjFactor']
                self.eod_data_dict['CloseToCloseReturn'] = self.price_df.shift(self.return_end_date, axis=1) / self.price_df.shift(self.return_start_date, axis=1) - 1
                self.eod_data_dict['CloseToCloseReturn'] = self.eod_data_dict['CloseToCloseReturn'] - self.eod_data_dict[
                    'CloseToCloseReturn'].mean()
                if not self.valid_y:
                    return self.eod_data_dict['CloseToCloseReturn']
            elif self.return_type == 'vwap_to_vwap':
                self.price_df = self.eod_data_dict['VWAP'] * self.eod_data_dict['AdjFactor']
                self.eod_data_dict['VwapToVwapReturn'] = self.price_df.shift(self.return_end_date, axis=1) / self.price_df.shift(self.return_start_date, axis=1) - 1
                self.eod_data_dict['VwapToVwapReturn'] = self.eod_data_dict['VwapToVwapReturn'] - self.eod_data_dict[
                    'VwapToVwapReturn'].mean()
                if not self.valid_y:
                    return self.eod_data_dict['VwapToVwapReturn']
            elif self.return_type == 'close_to_open':
                self.close_df = self.eod_data_dict['ClosePrice'] * self.eod_data_dict['AdjFactor']
                self.open_df = self.eod_data_dict['OpenPrice'] * self.eod_data_dict['AdjFactor']
                self.eod_data_dict['CloseToOpenReturn'] = self.open_df.shift(-1, axis=1) / self.close_df.shift(0, axis=1) - 1
                self.eod_data_dict['CloseToOpenReturn'] = self.eod_data_dict['CloseToOpenReturn'] - self.eod_data_dict[
                    'CloseToOpenReturn'].mean()
                if not self.valid_y:
                    return self.eod_data_dict['CloseToOpenReturn']

            if self.valid_y:
                self.get_valid_df()
                # 在这里先把跌停日的vwap删去，再用后一日vwap填充
                drop_down_limit_vwap = self.price_df * self.down_valid_df
                bfill_VWAP = (self.price_df - self.price_df) + drop_down_limit_vwap.bfill(axis=1)
                if self.return_type == "close_to_close":
                    self.return_df = bfill_VWAP.shift(self.return_end_date + 1, axis=1) / bfill_VWAP.shift(self.return_start_date + 1, axis=1) - 1
                    self.return_df = self.return_df - self.return_df.mean()
                elif self.return_type == "close_to_open":
                    self.get_valid_overnight_df()
                    drop_up_limit_close = self.close_df * self.up_feasible_stock_df
                    self.return_df = self.open_df.shift(-1, axis=1) / drop_up_limit_close.shift(0, axis=1) - 1
                    self.return_df = self.return_df - self.return_df.mean()
                else:
                    self.return_df = bfill_VWAP.shift(self.return_end_date, axis=1) / bfill_VWAP.shift(self.return_start_date, axis=1) - 1
                    self.return_df = self.return_df - self.return_df.mean()
                self.y_mask = self.return_df.copy()
                self.y_mask.iloc[:, :] = np.where(self.return_df.isna(), np.nan, 1)
                return self.return_df
        else:
            raise NotImplementedError('no diy')
            self.return_df = self.myconnector.get_eod_feature(fields=[cfg.diy_return_name],
                                                       where=cfg.diy_return_path,
                                                       tickers=self.stock_pool,
                                                       dates=self.date_list)
            self.return_df = self.return_df[cfg.diy_return_name].to_dataframe()
            self.y_mask = self.return_df.copy()
            self.y_mask.iloc[:, :] = np.where(self.return_df.isna(), np.nan, 1)
            return self.return_df


    def get_sw_ind_df(self) -> pd.DataFrame:
        """ get sw class (申万一级行业分类) """
        # sw1_df = (self.myconnector.get_sw_members(level = level))[["index_code", "con_code"]]
        sw1_df = self.myconnector.get_sw_members().drop_duplicates(subset=['con_code'])

        inds = ["100000"] + list(set(sw1_df.index_code))
        ones = pd.Series(1, index=self.stock_pool)
        temp = pd.concat([ones, sw1_df.set_index("con_code")],
                         axis=1).reset_index().set_index(["index"]).loc[
            self.stock_pool].reset_index().set_index(["index", "index_code"])
        ind_df = pd.DataFrame(temp.unstack().fillna(0).values,
                              index=self.stock_pool, columns=inds).T
        return ind_df

    def get_valid_df(self):

        # if fix stock pool
        self.feasible_stocks_df = self.get_issued_stock_df()
        self.eod_data_dict['feasible_stocks_df'] = self.feasible_stocks_df

        # suspend
        self.suspend_df = self.get_suspend()

        # buy limit 
        self.up_valid_df = self.get_status(up_down=1)
        self.up_feasible_stock_df = self.up_valid_df * self.feasible_stocks_df * self.suspend_df

        # sell limit 
        self.down_valid_df = self.get_status(up_down=-1)
        self.down_feasible_stock_df = self.down_valid_df * self.feasible_stocks_df

        # feasible_stock_df shift forward
        self.up_feasible_stock_df = self.up_feasible_stock_df.shift(-1,axis=1)
        self.down_feasible_stock_df = self.down_feasible_stock_df

    def get_valid_overnight_df(self):
        """ overnight valid """
        self.feasible_stocks_df = self.get_issued_stock_df()
        self.eod_data_dict['feasible_stocks_df'] = self.feasible_stocks_df

        # suspend
        self.suspend_df = self.get_suspend()

        # buy limit 
        self.up_valid_df = (self.eod_data_dict['ClosePrice'] <= self.eod_data_dict['UpLimitPrice'] - 0.01)
        self.up_valid_df.iloc[:, :] = np.where(self.up_valid_df == 0, np.nan, 1)
        self.up_feasible_stock_df = self.up_valid_df * self.feasible_stocks_df * self.suspend_df

        # sell limit 
        self.down_valid_df = self.get_status(up_down=-1)
        self.down_feasible_stock_df = self.down_valid_df * self.feasible_stocks_df

        # feasible_stock_df shift forward 
        self.up_feasible_stock_df = self.up_feasible_stock_df
        self.down_feasible_stock_df = self.down_feasible_stock_df

    def get_issued_stock_df(self):
        """ get issued stock """
        # ticker_basic_df = self.myconnector.get_ticker_basic(tickers=self.stock_pool, source='stock')
        # issue_date_dict = ticker_basic_df['listDate'].to_dict()
        issue_date_dict = self.myconnector.get_ticker_list_date()
        issue_status_df = pd.DataFrame(
            np.ones_like(self.eod_data_dict['ClosePrice']),
            index=self.eod_data_dict['ClosePrice'].index,
            columns=self.eod_data_dict['ClosePrice'].columns
        ).T
        for ticker in self.stock_pool:
            issue_status_df[ticker].loc[:self.get_previous_N_tradedate(issue_date_dict[ticker], -60)] = np.nan
        return issue_status_df.T

    def get_suspend(self):
        """ determine if suspended """
        # 停牌的处理：过去10天有停牌的不买
        suspend_df = 1 * (self.eod_data_dict['SuspendStatus'].rolling(10, axis=1).sum() == 0)
        for window in range(10):
            suspend_df.iloc[:, window] = 1 * (self.eod_data_dict['SuspendStatus'].rolling(window, axis=1).sum() == 0).iloc[:, window]
        suspend_df = suspend_df.replace(0, np.nan)
        self.eod_data_dict['suspend_df'] = suspend_df.astype('float64')
        return suspend_df

    def get_previous_N_tradedate(self, date, N=1):
        """
        get n trade dates before the current date 
        :param date: has to be a trade date
        :param N:
        :return:
        """
        try:
            pre_date = str(self.calendar[np.maximum(self.calendar.index(int(date)) - N, 0)])
        except:
            pre_date = date
        return pre_date

    def get_status(self, up_down):
        """
        TradeStatus/STStatus/UpDownLimitStatus
        """

        status_df = pd.DataFrame()

        if up_down == 1:
            status_df = (((self.eod_data_dict['STStatus'] == 0).mul(1)
                            # + (self.get_up_down_limit() != 1).mul(1)
                            + (self.eod_data_dict['OpenPrice'] != self.eod_data_dict['UpLimitPrice']).mul(1)
                            + (self.eod_data_dict['TradeStatus'] == 0).mul(1)
                            + (self.eod_data_dict['SuspendStatus'] == 0).mul(1)
                            + (self.eod_data_dict['IssueStatus'] == 0).mul(1)) == 5) * 1
            status_df = status_df.replace(0, np.nan)
            self.eod_data_dict['status_up_df'] = status_df

        elif up_down == -1:
            status_df = (((self.eod_data_dict['STStatus'] == 0).mul(1)
                            # + (self.get_up_down_limit() != -1).mul(1)
                            + (self.eod_data_dict['OpenPrice'] != self.eod_data_dict['DownLimitPrice']).mul(1)
                            + (self.eod_data_dict['TradeStatus'] == 0).mul(1))
                            + (self.eod_data_dict['SuspendStatus'] == 0).mul(1)
                            + (self.eod_data_dict['IssueStatus'] == 0).mul(1) == 5) * 1
            status_df = status_df.replace(0, np.nan)
            self.eod_data_dict['status_down_df'] = status_df

        return status_df

    def get_up_down_limit(self):
        """
        一字涨跌停的flag，涨停为1，跌停为-1，其余为0
        注意：没有上市的股票也被fill成了0，因此对股票是否上市交易没有判断作用
        """
        limit_df = 1 * (self.eod_data_dict["ClosePrice"] == self.eod_data_dict["OpenPrice"]) * (
                self.eod_data_dict["ClosePrice"] == self.eod_data_dict["HighestPrice"]) * (
                           self.eod_data_dict["ClosePrice"] == self.eod_data_dict["LowestPrice"])
        day_return = self.eod_data_dict['ClosePrice'] / self.eod_data_dict['PreClosePrice'] - 1
        direction_df = (day_return / day_return.abs()).fillna(0)
        return (limit_df * direction_df).fillna(0)

    # def get_close_up_limit(self):
    #     '''
    #     判断收盘是否是涨停
    #     :return:
    #     '''
    #     kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] == "688"]
    #     not_kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] != "688"]
    #     up_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1,axis=1)) > 1.099
    #     kc_up_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1,axis=1)) > 1.199
    #     up_limit[kc_index] = 0
    #     kc_up_limit[not_kc_index] = 0
    #     close_up_limit = up_limit + kc_up_limit
    #     return close_up_limit

    # def get_close_down_limit(self):
    #     '''
    #     判断收盘是否是跌停
    #     :return:
    #     '''
    #     kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] == "688"]
    #     not_kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] != "688"]
    #     down_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1,axis=1)) < 0.901
    #     kc_down_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1,axis=1)) < 0.801
    #     down_limit[kc_index] = 0
    #     kc_down_limit[not_kc_index] = 0
    #     close_down_limit = down_limit + kc_down_limit
    #     return close_down_limit

    '''
    ———————————————————————— prepare X and y in ML ————————————————————————————
    '''

    def prepare_data(self, train_date_list, test_date_list):
        """
        convert factor dict to a 2d dataframe to get train and test X 
        dfs will be doubly indexed by tickers and date
        """
        # train test split 
        self.train_data = pd.DataFrame(
            index=self.return_data[train_date_list[self.lag_date - 1:]].stack(dropna=False).index,
            columns=self.features + ['label'], 
            dtype='float'
        )
        self.test_data = pd.DataFrame(
            index=self.return_data[test_date_list[self.lag_date - 1:]].stack(dropna=False).index,
            columns=self.features + ['label'], 
            dtype='float'
        )

        # populate
        for feature in self.features:
            feature_name = feature[:-2]  # remove lag
            lag = int(feature[-1])
            self.train_data[feature] = self.factor_dict[feature_name][train_date_list[self.lag_date - 1 - lag:len(train_date_list) - lag]].stack(dropna=False).values
            self.test_data[feature] = self.factor_dict[feature_name][test_date_list[self.lag_date - 1 - lag:len(test_date_list) - lag]].stack(dropna=False).values
        self.train_data['label'] = self.return_data[train_date_list[self.lag_date - 1:]].stack(dropna=False).values
        self.test_data['label'] = self.return_data[test_date_list[self.lag_date - 1:]].stack(dropna=False).values
        # self.train_mask = self.y_mask[train_date_list[self.lag_date - 1:]].stack(dropna=False).values
        self.test_mask = self.y_mask[test_date_list[self.lag_date - 1:]].stack(dropna=False).values
        self.test_orig_Y = self.test_data['label'].copy()
        self.train_orig_Y = self.train_data['label'].copy()
        # print(self.train_data)

    def standard_clean_data(self):
        """
        preprocess and standardize train_X, train_Y, test_X, test_Y
        """
        if self.vague:
            self.vague_zero(threshold=cfg.vague_threshold)
            self.vague_large(rangerate=cfg.vague_rangerate)
        if 'Class' in self.type:
            self.get_class_label_y()
        self.split_data()  # take labels out 
        self.winsorize()   # remove extreme values 
        self.delete_nan()  # remove nan if too much 
        self.nanprocess()  # 
        self.clean_y_nan()
        if 'Class' in self.type:
            self.train_Y = self.train_Y.astype(int)

        return self.train_X, self.train_Y, self.test_X, self.test_Y

    def split_data(self):
        """ take out labels """
        self.train_X = self.train_data.drop(['label'], axis=1)
        self.train_Y = self.train_data['label']
        self.test_X = self.test_data.drop(['label'], axis=1)
        self.test_Y = self.test_data['label']

    def vague_zero(self, threshold=0.01):
        """ assign nan to y values that are close to 0 by a predefined threshold """
        print('Train Vague Zero Ratio: {}%'.format(round(np.sum(self.train_data['label'].abs() < threshold) / len(self.train_data['label'].dropna()) * 100, 3)))
        self.train_data['label'] = np.where(self.train_data['label'].abs() < threshold, np.nan, self.train_data['label'])

        # print('Test Vague Ratio: {}'.format(np.sum(self.test_data['label'].abs() < 0.01) / len(self.test_data['label'])))
        # self.test_data['label'] = np.where(self.test_data['label'].abs() < 0.01, np.nan, self.test_data['label'])

    def vague_large(self, rangerate = 1.5):
        """ assign nan to y values that are too large (abnormal data) """
        print('Train Vague Large Ratio: {}%'.format(round(np.sum((self.train_data['label'] - self.train_data['label'].mean()).abs() > rangerate * self.train_data['label'].std()) / len(self.train_data['label'].dropna())* 100, 3)))
        print('Pos Large Y Value:{}'.format(round(self.train_data['label'].mean() + rangerate * self.train_data['label'].std(), 4)))
        print('Neg Large Y Value:{}\n'.format(round(self.train_data['label'].mean() - rangerate * self.train_data['label'].std(), 4)))
        self.train_data['label'] = np.where((self.train_data['label'] - self.train_data['label'].mean()).abs() > rangerate * self.train_data['label'].std(), np.nan, self.train_data['label'])

    def get_class_label_y(self):
        """
        get y for classification. Regression would not undergo this transformation 
        """
        self.train_data['label'] = (self.train_data['label'] > 0) + (self.train_data['label'] - self.train_data['label'])
        self.test_data['label'] = (self.test_data['label'] > 0) + (self.test_data['label'] - self.test_data['label'])

    def winsorize(self):
        """ 
        empirical adjustments to factor values. The procedure is as follows: 

        intermediate range = Q90 - Q10 

        change values 4 times away from the median to be nan 
        change values between 2.5 - 4 times away from the median to be 2.5 times away 
        """
        self.train_X.iloc[:, :] = np.where(np.isinf(self.train_X.values), np.nan, self.train_X.values)

        QT90 = np.nanquantile(self.train_X, 0.9, axis=0)
        QT50 = np.nanquantile(self.train_X, 0.5, axis=0)
        QT10 = np.nanquantile(self.train_X, 0.1, axis=0)

        UP = QT50 + 2.5 * (QT90 - QT10)
        DOWN = QT50 - 2.5 * (QT90 - QT10)
        ExUP = QT50 + 4 * (QT90 - QT10)
        ExDOWN = QT50 - 4 * (QT90 - QT10)

        if self.skip_features is not None:
            skip_features = list(set(self.skip_features))
            skip_idx = [self.features.index(x) for x in skip_features if x in self.features]
            keepValues = self.train_X[skip_features[skip_idx]].copy(deep = True)

        # assign nan to too large/small a value 
        self.train_X.iloc[:, :] = np.where(self.train_X > ExUP, np.nan, self.train_X)
        self.train_X.iloc[:, :] = np.where(self.train_X < ExDOWN, np.nan, self.train_X)
        # winsorize
        self.train_X.iloc[:, :] = np.where(self.train_X > UP, UP, self.train_X)
        self.train_X.iloc[:, :] = np.where(self.train_X < DOWN, DOWN, self.train_X)
        self.test_X.iloc[:, :] = np.where(self.test_X > UP, UP, self.test_X)
        self.test_X.iloc[:, :] = np.where(self.test_X < DOWN, DOWN, self.test_X)
        cutted_UP_X = np.sum((self.train_X.values > UP - 1e-7), axis=0) / self.train_X.values.shape[0]
        cutted_DOWN_X = np.sum((self.train_X.values < DOWN + 1e-7), axis=0) / self.train_X.values.shape[0]

        if self.skip_features is not None:
            cutted_UP_X[skip_idx] = np.nan
            cutted_DOWN_X[skip_idx] = np.nan
        # print(self.train_X, self.test_X)
        # print(cutted_UP_X)
        # print(cutted_DOWN_X)
        print(f"Max cutted_UP_X = {np.nanmax(cutted_UP_X)}, {self.features[np.nanargmax(cutted_UP_X)]}")
        print(f"Max cutted_DOWN_X = {np.nanmax(cutted_DOWN_X)}, {self.features[np.nanargmax(cutted_DOWN_X)]}")

        # normalization
        if self.whether_norm:
            self.train_X = (self.train_X - QT50) / (QT90 - QT10 + 1e-7)
            self.test_X = (self.test_X - QT50) / (QT90 - QT10 + 1e-7)

        # white list some factors (no processing for those factor values)
        if self.skip_features is not None:
            self.train_X[skip_features[skip_idx]] = keepValues

    def nanprocess(self):
        """ process nan, fill0 or ignore """
        if self.nanmethod == "fill0":
            self.train_X[np.isnan(self.train_X)] = 0
            self.test_X[np.isnan(self.test_X)] = 0
        elif self.nanmethod == "ignore":
            pass

    def delete_nan(self):
        """ empirical adjustment: take away samples that have too much nan (larger than 30% of factor values) """
        na_pct = np.sum(np.isnan(self.train_X), axis=1) / self.train_X.shape[1]
        cut_na_points = (na_pct > 0.3).values.reshape(self.train_Y.shape)
        self.train_Y.iloc[:] = np.where(cut_na_points, np.nan, self.train_Y)
        print(f'Deleted NA samples {round(np.sum(cut_na_points) / self.train_X.shape[0] * 100, 3)}%')
        print(f'Deleted NA samples {np.sum(cut_na_points)}')

        na_pct = np.sum(np.isnan(self.test_X), axis=1) / self.test_X.shape[1]
        cut_na_points = (na_pct > 0.3).values.reshape(self.test_Y.shape)
        self.test_Y.iloc[:] = np.where(cut_na_points, np.nan, self.test_Y)

    def clean_y_nan(self):
        """ remove corresponding y values """
        self.na_idx = np.argwhere(~np.isnan(self.train_Y.values))
        print(f'Deleted y-NA samples {round((1 - len(self.na_idx) / len(self.train_Y.values)) * 100, 3)}%')
        self.train_X = self.train_X.iloc[self.na_idx[:, 0]]
        self.train_Y = self.train_Y.iloc[self.na_idx[:, 0]]
        self.train_orig_Y_mask = self.train_orig_Y.iloc[self.na_idx[:, 0]]

    def weights_cal(self, Y, method, na_idx=None):
        """ assign weights """
        W = np.zeros(Y.shape)

        row_num, col_num = self.train_orig_Y.unstack().shape

        if method == 'EW':
            W = W + 1

        elif '+' in method:
            raise NotImplementedError('Other weightin Not yet implemented')
            Y_method, TD_method, ID_method = method.split('+')

            # 处理多头强Y值的权重
            if 'LY' in Y_method:
                Y_num = float(Y_method[2:])
                if 'Class' in self.type:
                    rankY = (rankdata(self.train_orig_Y_mask.values) - 1) / (self.train_orig_Y_mask.shape[0] - 1)
                else:
                    rankY = (rankdata(Y.values) - 1) / (Y.shape[0] - 1)

                # large Y
                big_Y = (rankY > 0.8)
                ly_W = np.where(big_Y, Y_num, 1)
            elif 'SY' in Y_method:
                Y_num = float(Y_method[2:])
                if 'Class' in self.type:
                    rankY = (rankdata(self.train_orig_Y_mask.values) - 1) / (self.train_orig_Y_mask.shape[0] - 1)
                else:
                    rankY = (rankdata(Y.values) - 1) / (Y.shape[0] - 1)

                # small Y
                big_Y = (rankY < 0.2)
                ly_W = np.where(big_Y, Y_num, 1)
            else:
                ly_W = W + 1

            # 处理时序上Y值的权重
            if 'TD' in TD_method:
                TD_num = int(TD_method[2:])
                ts_W = self.train_orig_Y.unstack().copy(deep=True)
                m = int(col_num / TD_num)
                # 线性衰减
                if 'linear' in cfg.decay_func:
                    for i in range(TD_num):
                        ts_W.iloc[:, i * m:(i + 1) * m] = (i + 1) / TD_num
                # 多项式衰减
                elif 'poly' in cfg.decay_func:
                    poly_deg = int(cfg.decay_func.split('+')[1])
                    for i in range(TD_num):
                        ts_W.iloc[:, i * m:(i + 1) * m] = 1 / (TD_num - i) ** poly_deg
                # 指数衰减
                elif 'exp' in cfg.decay_func:
                    for i in range(TD_num):
                        ts_W.iloc[:, i * m:(i + 1) * m] = 1 / np.exp(TD_num - i - 1)
                ts_W = ts_W.stack(dropna=None).iloc[na_idx[:, 0]].values
            else:
                ts_W = W + 1

            # 处理时序上指数的权重
            index_data = self.myconnector.get_eod_history(tickers=['000905'], start_date=self.start_date,
                                             end_date=self.end_date, source='index')
            index_ret = index_data['OpenPrice'].shift(-4, axis = 1) / index_data['OpenPrice'].shift(-1, axis = 1) - 1
            index_ret = index_ret[self.train_orig_Y.unstack().columns]
            id_W = self.train_orig_Y.unstack().copy(deep=True)

            if 'LID' in ID_method:
                ID_num = float(ID_method[3:])

                # large index
                id_W.iloc[:, :] = np.where(index_ret > 0, ID_num, 1)
                id_W = id_W.stack(dropna=None).iloc[na_idx[:, 0]].values
            elif 'SID' in ID_method:
                ID_num = float(ID_method[3:])

                # small index
                id_W.iloc[:, :] = np.where(index_ret < 0, ID_num, 1)
                id_W = id_W.stack(dropna=None).iloc[na_idx[:, 0]].values
            elif 'BID' in ID_method:
                ID_num = float(ID_method[3:])

                # both index
                id_W.iloc[:, :] = np.where((index_ret < -0.02) * (index_ret > 0.02), ID_num, 1)
                id_W = id_W.stack(dropna=None).iloc[na_idx[:, 0]].values
            else:
                id_W = W + 1

            W = ly_W * ts_W * id_W

        else:
            print('...Error:{}...'.format(method))

        return list(W.reshape(-1))

    '''
    ———————————————————————— io ————————————————————————————
    '''

    def save_factor_data(self, feature_name, eod_df, factor_type="ml_factors"):
        """
        save factors 
        """
        # feature_name = "eod_" + feature_name + int(self.neutralize) * '_neutralize'
        # self.myconnector.save_eod_feature(data={feature_name: eod_df},
        #                                   where=cfg.factor_save_path + "{}".format(factor_type),
        #                                   feature_type='eod', encrypt=False)
        self.myconnector.save_eod_feature(
            feature_name, 
            eod_df, 
            des='ml_factor'
        )
