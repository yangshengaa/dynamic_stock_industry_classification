"""
回测eod数据读入 及 基本处理
"""

# load packages 
import os
import sys
import traceback
import numpy as np
import pandas as pd

# load files 
from src.backtest.configuration import config as cfg
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

class DataAssist:

    def __init__(self, offline=False, eod_data_dict=None):
        self.index_list = cfg.index_list
        self.weight_by_fmv = cfg.weight_index_by_fmv
        self.use_independent_return_benchmark = cfg.use_independent_return_benchmark
        self.return_benchmark_index = cfg.return_benchmark_index
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        self.data_start_date = cfg.start_date
        self.data_end_date = cfg.end_date
        self.return_type = cfg.return_type
        self.cost = cfg.cost
        self.offline = offline
        self.max_lookback = cfg.max_lookback
        self.max_lookforward = cfg.max_lookforward

        if offline:
            self.eod_data_dict = eod_data_dict
            self.stock_pool = list(self.eod_data_dict["ClosePrice"].index)
            self.calendar = [str(x) for x in self.eod_data_dict['calendar']]
        else: 
            # self.myconnector = PqiDataSdk(user=cfg.user, size=cfg.ds_max_processes, pool_type="mp", log=False,
            #                               offline=True)
            self.myconnector = PqiDataSdkOffline()
            self.stock_pool = self.myconnector.get_ticker_list()  # 先读入所有股票，后再加mask
            self.get_data_date_range()
            self.get_eod_data()
            self.ind_df = self.get_sw_ind_df()
            for k in self.eod_data_dict.keys():
                self.eod_data_dict[k].columns = [str(x) for x in self.eod_data_dict[k].columns]
            self.eod_data_dict["ind_df"] = self.ind_df
            self.calendar = np.array(sorted([int(x) for x in self.myconnector.select_trade_dates(start_date='20010101',end_date='21000101')]))
            self.eod_data_dict["calendar"] = self.calendar

        # 未赋值变量
        self.depth_data_dict = {}
        self.trade_data_dict = {}
        self.order_data_dict = {}
        self.ret_df = pd.DataFrame()
        self.up_valid_df = pd.DataFrame()
        self.down_valid_df = pd.DataFrame()
        self.index_ret_series = pd.Series()
        self.up_feasible_stock_df = pd.DataFrame()
        self.down_feasible_stock_df = pd.DataFrame()

    # TODO: 修改指数成分股的读取方式
    # @staticmethod
    # def get_stock_weight(index):
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
    #     eod_path = cfg.index_member_stock_path 
    #     index_file = index_to_weight[index]

    #     # 读入并转成dataframe形式
    #     index_data = np.load(os.path.join(eod_path, index_file))
    #     tickers = np.load(os.path.join(eod_path, 'ticker_names.npy'))
    #     dates = np.load(os.path.join(eod_path, 'dates.npy'))
    #     df = pd.DataFrame(index_data, columns=dates, index=tickers)
        
    #     # 与设置的时间段对齐
    #     df_selected = df.loc[:, cfg.trade_dates]
    #     return df_selected
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


    @staticmethod
    def get_index_mask(index_list):
        """
        读取指数成分股权重，转换成指数成分股mask: 当天某股票为该指数成分股为1，否则为nan
        :param index_list: 一个装有指数的列表，支持组合指数
        :return 指数mask
        """
        agg_index_mask = False
        # 读入和拼接多个index
        for index in index_list:
            index_weight = DataAssist.get_stock_weight(index)
            index_mask = index_weight.notna()
            agg_index_mask = agg_index_mask | index_mask
        # 将dataframe转换为1和nan的矩阵
        agg_index_mask = agg_index_mask.astype(int) / agg_index_mask.astype(int)
        return agg_index_mask        

    # TODO: 更改eod data 读取方式
    def get_eod_data(self):
        price_dict = self.myconnector.get_eod_history(
            tickers=self.stock_pool,
            start_date=self.data_start_date,
            end_date=self.data_end_date, 
            source="stock"
        )

        self.eod_data_dict = price_dict
        std_shape = self.eod_data_dict["ClosePrice"].shape

        # 检查eod_data_dict中是否所有字段都是同一个长度
        for k in self.eod_data_dict.keys():
            if self.eod_data_dict[k].shape != std_shape:
                print(k + " has special shape")
                print(self.eod_data_dict[k].shape)

    def get_return_data(self):
        """
        得到不同版本的return数据
        增加一个vwap
        这里对涨跌停的处理有点问题，需要再优化
        TODO: close_to_close和open_to_open目前是失真的，涨跌停没有很好地处理
        """

        price_df = pd.DataFrame()
        if self.return_type == 'open_to_open':
            price_df = self.eod_data_dict['OpenPrice'] * self.eod_data_dict['AdjFactor']
            self.eod_data_dict["OpenToOpenReturn"] = price_df.shift(-2, axis=1) / price_df.shift(-1, axis=1) - 1
        elif self.return_type == 'close_to_close':
            raise NotImplementedError
            price_df = self.eod_data_dict['ClosePrice'] * self.eod_data_dict['AdjFactor']
            self.eod_data_dict['CloseToCloseReturn'] = price_df.shift(-2, axis=1) / price_df.shift(-1, axis=1) - 1
        elif self.return_type == 'vwap_to_vwap':
            price_df = self.eod_data_dict['VWAP'] * self.eod_data_dict['AdjFactor']
            self.eod_data_dict['VwapToVwapReturn'] = price_df.shift(-2, axis=1) / price_df.shift(-1, axis=1) - 1

        self.ret_df = price_df.shift(-2, axis=1) / price_df.shift(-1, axis=1) - 1

        # 获取测试掩码
        self.get_valid_df()
        # 在这里先把跌停日的vwap删去，再用后一日vwap填充
        drop_down_limit_vwap = price_df * self.down_feasible_stock_df
        bfill_VWAP = (price_df - price_df) + drop_down_limit_vwap.bfill(axis=1)
        if self.return_type == "close_to_close":
            self.ret_df = bfill_VWAP.shift(-1, axis=1) / bfill_VWAP.shift(0, axis=1) - 1
        else:
            self.ret_df = bfill_VWAP.shift(-2, axis=1) / bfill_VWAP.shift(-1, axis=1) - 1
            self.ret_n_df = bfill_VWAP.shift(-1-cfg.adj_freq,axis=1) / bfill_VWAP.shift(-1, axis=1) - 1

        # 粗糙修改
        if self.offline:
            self.ret_df = self.eod_data_dict['ret_df']
            self.ret_n_df = self.eod_data_dict['ret_n_df']
        else:
            # ret_df的存储
            self.eod_data_dict['ret_df'] = self.ret_df
            self.eod_data_dict['ret_n_df'] = self.ret_n_df

        # get vwap decay return
        self.decay_ret_df_list = []
        for i in range(cfg.ic_decay):
            vwap_decay_ret_df = bfill_VWAP.shift(-i-2,axis=1)/bfill_VWAP.shift(-1,axis=1) - 1
            self.decay_ret_df_list.append(vwap_decay_ret_df)

        # get split return df
        self.split_ret_dict = {}
        price_close_0 = (self.eod_data_dict['ClosePrice']*self.eod_data_dict['AdjFactor']).shift(1,axis=1)
        price_open_1 = self.eod_data_dict['OpenPrice']*self.eod_data_dict['AdjFactor']
        price_vwap_1 = self.eod_data_dict['VWAP']*self.eod_data_dict['AdjFactor']
        price_close_1 = self.eod_data_dict['ClosePrice']*self.eod_data_dict['AdjFactor']
        price_open_2 = (self.eod_data_dict['OpenPrice']*self.eod_data_dict['AdjFactor']).shift(-1,axis=1)
        price_vwap_2 = (self.eod_data_dict['VWAP']*self.eod_data_dict['AdjFactor']).shift(-1,axis=1)
        self.split_ret_type_list = ['cto','otv','vtc','cto1','o1tv1']
        self.split_ret_dict["cto"] = price_open_1/price_close_0 - 1
        self.split_ret_dict['otv'] = price_vwap_1/price_close_0 - 1
        self.split_ret_dict['vtc'] = price_close_1/price_close_0 - 1
        self.split_ret_dict['cto1'] = price_open_2/price_close_0 - 1
        self.split_ret_dict['o1tv1'] = price_vwap_2/price_close_0 - 1

        ## 获取指数数据
        self.get_index_ret(weight=self.weight_by_fmv)

        ## 删除lookbackh和lookforward的部分
        # # ! Update: ignore lookback and lookforward
        # if not self.offline:
        #     self.change_data_range()

    # TODO: 更改获取上市股票
    def get_issued_stock_df(self):
        '''
        获取上市股票 (不交易上市未满60天的新股)
        :return:
        '''
        # ticker_basic_df = self.myconnector.get_ticker_basic(tickers=self.stock_pool, source='stock')
        # issue_date_dict = ticker_basic_df['listDate'].to_dict()
        issue_date_dict = self.myconnector.get_ticker_list_date()
        issue_status_df = pd.DataFrame(np.ones_like(self.eod_data_dict['ClosePrice']),
                                       index=self.eod_data_dict['ClosePrice'].index,
                                       columns=self.eod_data_dict['ClosePrice'].columns).T
        for ticker in self.stock_pool:
            # ! the following is a very dangerous operation. Revise later
            issue_status_df[ticker].loc[:self.get_previous_N_tradedate(issue_date_dict[ticker], -60)] = np.nan
        return issue_status_df.T

    def get_valid_df(self):

        if self.offline:
            self.feasible_stocks_df = self.eod_data_dict['feasible_stocks_df']
            self.index_mask = self.eod_data_dict['index_mask']
        else:
            self.feasible_stocks_df = self.get_issued_stock_df()
            self.eod_data_dict['feasible_stocks_df'] = self.feasible_stocks_df

            # 获取mask
            temp_index_mask = self.get_index_mask(self.index_list)
            temp_index_mask = (self.eod_data_dict['ClosePrice'] - self.eod_data_dict['ClosePrice']) +  temp_index_mask # 对齐index，方便放入shm
            self.eod_data_dict['index_mask'] = temp_index_mask
            

        # 停牌限制
        self.suspend_df = self.get_suspend()

        # 买入限制
        self.up_valid_df = self.get_status(up_down=1)
        self.up_feasible_stock_df = self.up_valid_df * self.feasible_stocks_df * self.suspend_df

        # 跌停处理
        self.down_valid_df = self.get_status(up_down=-1)
        self.down_feasible_stock_df = self.down_valid_df * self.feasible_stocks_df

        # feasible_stock_df往后挪一天
        self.up_feasible_stock_df = self.up_feasible_stock_df.shift(-1, axis=1)  
        self.down_feasible_stock_df = self.down_feasible_stock_df


    # TODO: 
    def get_suspend(self):
        '''
        判断停牌
        :return:
        '''
        if self.offline:
            return self.eod_data_dict['suspend_df']
        else:
            # 停牌的处理：过去10天有停牌的不买
            suspend_df = 1 * (self.eod_data_dict['SuspendStatus'].rolling(10, axis=1).sum() == 0)
            suspend_df.iloc[:, :] = np.where(suspend_df == 0, np.nan, suspend_df)
            self.eod_data_dict['suspend_df'] = suspend_df.astype('float64')
        return suspend_df


    def get_previous_N_tradedate(self, date, N=1):
        """
        get trade days N days ago 
        :param date: must be a trade date
        :param N:
        :return:
        """
        # find index
        cur_date_idx = np.where(self.calendar == int(date))[0]
        # if not in the current calendar, the solution is imperfect: just return itself.
        if len(cur_date_idx) == 0:
            return date
        
        prev_date_idx = cur_date_idx[0] - N 
        # put in range
        prev_date_idx_aligned = max(0, min(len(self.calendar) - 1, prev_date_idx)) 
        # select date
        pre_date = str(self.calendar[prev_date_idx_aligned])
        return pre_date

    # TODO: 
    def get_status(self, up_down):
        """
        TradeStatus/STStatus/UpDownLimitStatus
        :return:
        """
        # 新股的处理：上市60天以上

        status_df = pd.DataFrame()
        if self.offline:
            if up_down == 1:
                return self.eod_data_dict['status_up_df']
            elif up_down == -1:
                return self.eod_data_dict['status_down_df']
        else:
            if up_down == 1:
                status_df = (((self.eod_data_dict['STStatus'] == 0).mul(1)
                              # + (self.get_up_down_limit() != 1).mul(1)
                              + ((self.eod_data_dict['UpLimitPrice'] - self.eod_data_dict['OpenPrice']) > 1e-6).mul(1)  
                              + (self.eod_data_dict['TradeStatus'] == 0).mul(1)
                              + (self.eod_data_dict['SuspendStatus'] == 0).mul(1)
                              + (self.eod_data_dict['IssueStatus'] == 0).mul(1)) == 5) * 1
                status_df = status_df.replace(0, np.nan)
                self.eod_data_dict['status_up_df'] = status_df

            elif up_down == -1:
                status_df = (((self.eod_data_dict['STStatus'] == 0).mul(1)
                              # + (self.get_up_down_limit() != -1).mul(1)
                              + ((self.eod_data_dict['OpenPrice'] - self.eod_data_dict['DownLimitPrice']) > 1e-6).mul(1)  
                              + (self.eod_data_dict['TradeStatus'] == 0).mul(1)) 
                              + (self.eod_data_dict['SuspendStatus'] == 0).mul(1)
                              + (self.eod_data_dict['IssueStatus'] == 0).mul(1) == 5) * 1
                status_df = status_df.replace(0, np.nan)
                self.eod_data_dict['status_down_df'] = status_df

        return status_df

    def get_up_down_limit(self):
        """
        一字涨跌停的flag，涨停为1，跌停为-1，其余为0
        注意: 没有上市的股票也被fill成了0，因此对股票是否上市交易没有判断作用
        """
        limit_df = 1 * (self.eod_data_dict["ClosePrice"] == self.eod_data_dict["OpenPrice"]) * (
                self.eod_data_dict["ClosePrice"] == self.eod_data_dict["HighestPrice"]) * (
                           self.eod_data_dict["ClosePrice"] == self.eod_data_dict["LowestPrice"])
        day_return = self.eod_data_dict['ClosePrice'] / self.eod_data_dict['PreClosePrice'] - 1
        direction_df = (day_return / day_return.abs()).fillna(0)
        return (limit_df * direction_df).fillna(0)

    def get_close_up_limit(self):
        '''
        判断收盘是否是涨停
        :return:
        '''
        kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] == "688"]
        not_kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] != "688"]
        up_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1, axis=1)) > 1.099
        kc_up_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1, axis=1)) > 1.199
        up_limit[kc_index] = 0
        kc_up_limit[not_kc_index] = 0
        close_up_limit = up_limit + kc_up_limit
        return close_up_limit

    def get_close_down_limit(self):
        '''
        判断收盘是否是跌停
        :return:
        '''
        kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] == "688"]
        not_kc_index = [index for index in self.eod_data_dict["ClosePrice"].index if index[:3] != "688"]
        down_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1,axis=1)) < 0.901
        kc_down_limit = 1*(self.eod_data_dict["ClosePrice"]/self.eod_data_dict["ClosePrice"].shift(1,axis=1)) < 0.801
        down_limit[kc_index] = 0
        kc_down_limit[not_kc_index] = 0
        close_down_limit = down_limit + kc_down_limit
        return close_down_limit

    # TODO: 更改指数收益率计算/读取方式
    # TODO: simplify structure
    def get_index_ret(self, weight=True):
        """
        根据票池计算指数
        - weight == True时
            - 如果只有一个index, 直接读取指数的oto收益率序列
            - 如果多与一个index, 用fmv撮合指数收益序列
        - weight == False时, 不管有多少个index都是等权计算
        """
        # 如果用独立的收益率序列
        if self.use_independent_return_benchmark:
            if not self.offline:  # 如果是第一次读取，则直接读取收益序列
                index_pool = cfg.index_dict[self.return_benchmark_index]
                self.index_data = self.myconnector.get_eod_history(
                    tickers=[index_pool], 
                    start_date=self.start_date,
                    end_date=self.end_date, 
                    source='index'
                )
                self.eod_data_dict['index_data'] = pd.concat(
                    [self.index_data['ClosePrice'], self.index_data['OpenPrice'],
                     self.index_data['HighestPrice'], self.index_data['LowestPrice'],
                     self.index_data['TradeValue'], self.index_data['TradeVolume'],
                     self.index_data['PreClosePrice']]
                )
                self.eod_data_dict['index_data'].index = ['0', '1', '2', '3', '4', '5', '6']  # 方便存入shm

            # ['ClosePrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'TradeValue', 'TradeVolume', 'PreClosePrice'])
            # [0, 1, 2, 3, 4, 5, 6] fyi
            index_open = self.eod_data_dict['index_data'].loc['1']
            self.index_ret_series = (index_open.shift(-2) / index_open.shift(-1) - 1)

        # 如果基准和交易票池相同，且单一指数，直接读取指数序列
        elif len(self.index_list) == 1 and weight:
            if not self.offline:  # 如果是第一次读取，则直接读取收益序列
                index_pool = cfg.index_dict[self.index_list[0]]
                self.index_data = self.myconnector.get_eod_history(
                    tickers=[index_pool], 
                    start_date=self.start_date,
                    end_date=self.end_date, 
                    source='index'
                )
                self.eod_data_dict['index_data'] = pd.concat(
                    [self.index_data['ClosePrice'], self.index_data['OpenPrice'],
                     self.index_data['HighestPrice'], self.index_data['LowestPrice'],
                     self.index_data['TradeValue'], self.index_data['TradeVolume'],
                     self.index_data['PreClosePrice']]
                )
                self.eod_data_dict['index_data'].index = ['0', '1', '2', '3', '4', '5', '6']  # 方便存入shm

            # ['ClosePrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'TradeValue', 'TradeVolume', 'PreClosePrice'])
            # [0, 1, 2, 3, 4, 5, 6] fyi
            index_open = self.eod_data_dict['index_data'].loc['1']
            self.index_ret_series = (index_open.shift(-2) / index_open.shift(-1) - 1)
        
        # 如果用复合指数，用可选票的fmv加权算收益率
        else:
            price_df = pd.DataFrame()
            if self.return_type == 'open_to_open':
                price_df = self.eod_data_dict['OpenPrice'] * self.eod_data_dict['AdjFactor']
            elif self.return_type == 'vwap_to_vwap':
                price_df = self.eod_data_dict['VWAP'] * self.eod_data_dict['AdjFactor']
            
            # TODO: 清除不必要的代码，如下一行
            self.eod_data_dict['index_data'] = price_df  # 这一行是没有用但必须的，因为自建指数的时候是不通过‘index_data’算的，但batch需要有它
            return_df =  (price_df.shift(-2,axis=1) / price_df.shift(-1,axis=1) - 1)  * self.eod_data_dict['index_mask'] 

            if weight:
                fmv_masked = self.eod_data_dict['FloatMarketValue'] * (return_df / return_df)  # 给fmv加掩码
                stock_weight_df = fmv_masked / fmv_masked.sum()                                # 掩码后计算权重
                index_weighted_ret_df = stock_weight_df * return_df                            
                self.index_ret_series = index_weighted_ret_df.sum()
            else:
                self.index_ret_series = return_df.mean(axis=0)

    # TODO: 更改行业读取方式
    def get_sw_ind_df(self, level=1):
        # sw1_df = (self.myconnector.get_sw_members(level=level))[["index_code", "con_code"]]
        # read and drop duplicates
        sw1_df = self.myconnector.get_sw_members().drop_duplicates(subset=['con_code'])
    
        inds = ["100000"] + list(set(sw1_df.index_code))
        ones = pd.Series(1, index=self.stock_pool)
        temp = pd.concat(
            [ones, sw1_df.set_index("con_code")], axis=1
        ).reset_index().set_index(["index"]).loc[
            self.stock_pool
        ].reset_index().set_index(["index", "index_code"])
        ind_df = pd.DataFrame(temp.unstack().fillna(0).values,
                              index=self.stock_pool, columns=inds).T
        return ind_df

    # TODO: 更改lookback接口
    def get_data_date_range(self):
        # # 数据向前读取max_lookback的长度
        # current_trade_date = self.start_date
        # for i in range(self.max_lookback):
        #     prev_trade_date = self.myconnector.get_prev_trade_date(trade_date=current_trade_date)
        #     if prev_trade_date == "":
        #         break
        #     else:
        #         current_trade_date = prev_trade_date
        # self.data_start_date = current_trade_date

        # # 数据向后读取三天，为了计算收益率
        # current_trade_date = self.end_date
        # for i in range(self.max_lookforward):
        #     next_trade_date = self.myconnector.get_next_trade_date(trade_date=current_trade_date)
        #     if next_trade_date is None:
        #         break
        #     else:
        #         current_trade_date = next_trade_date
        # self.data_end_date = current_trade_date
        self.data_start_date = self.start_date
        self.data_end_date = self.end_date

    def change_data_range(self):
        '''
        把lookback和lookforward的部分砍掉
        :return:
        '''
        date_list = self.ret_df.columns[self.max_lookback:-self.max_lookforward]
        self.ret_df = self.ret_df[date_list]
        self.ret_n_df = self.ret_n_df[date_list]
        self.up_feasible_stock_df = self.up_feasible_stock_df[date_list]
        self.down_feasible_stock_df = self.down_feasible_stock_df[date_list]
        self.index_ret_series = self.index_ret_series[date_list]

        for key in self.split_ret_dict.keys():
            self.split_ret_dict[key] = self.split_ret_dict[key][date_list]
        for i in range(len(self.decay_ret_df_list)):
            self.decay_ret_df_list[i] = self.decay_ret_df_list[i][date_list]
        for k in self.eod_data_dict.keys():
            if k != 'ind_df' and k != 'calendar' and k != 'index_data':
                self.eod_data_dict[k] = self.eod_data_dict[k][date_list]
