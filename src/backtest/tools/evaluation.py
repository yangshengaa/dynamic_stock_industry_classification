"""
绩效评估函数
"""

# load packages 
import sys
import copy
import warnings
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# sys.path.append("..")

# load files 
from src.backtest.configuration import config as cfg

# :param data_dict:
#             keys: long_ret_series, short_ret_series, group_ret_series,index_ret_series,
#                   IC,turnover,turnover_long,turnover_short,summary

class Evaluator:
    """
    Evaluator类，测试共分三种类型，IC测试、多空测试与分层回测
    """

    def __init__(self, data_assist, factor_df, factor_name):
        self.DataAssist = data_assist
        self.factor_df = factor_df
        self.factor_name = factor_name
        self.ret_df = self.DataAssist.ret_df
        self.ret_n_df = self.DataAssist.ret_n_df
        self.factor_df = self.factor_df.loc[self.ret_df.index]
        self.mask = np.where(np.isnan(self.factor_df), np.nan, 1)
        self.cost = self.DataAssist.cost
        self.stock_pool = self.DataAssist.stock_pool
        self.up_feasible_stock_df = self.DataAssist.up_feasible_stock_df
        self.down_feasible_stock_df = self.DataAssist.down_feasible_stock_df
        self.head = cfg.head
        self.freq = cfg.freq  # "W"表示按周调仓，否则按日调仓
        self.decay_list = cfg.decay_list
        self.lead_lag_list = cfg.lead_lag_list
        if self.freq == "W":
            self.adj_freq = 5
        else:
            self.adj_freq = cfg.adj_freq  # 按n日调仓模式下
        # self.by_days = cfg.by_days  # 按日调仓或每周周几调仓
        self.group_num = cfg.group_num

        # 未实际赋值
        self.ic_list = []
        self.rank_ic_list = []
        self.ic_stats_dict = {}
        self.rank_ic_stats_dict = {}
        self.ic_decay = []
        self.rank_ic_decay = []
        self.data_dict = {}
        self.summary = {}
        self.group_ret_series_list = []
        self.long_short_raw_res_dict = {}
        self.long_batch_signal_df_list = []
        self.short_batch_signal_df_list = []
        self.long_batch_avg_ret_list = []
        self.short_batch_avg_ret_list = []
        self.long_short_batch_avg_ret_list = []
        self.group_raw_res_dict = {}
        self.holding_stats_df = pd.DataFrame()
        self.pool_stats_dict = {}

        self.need_swapping = False  # 多空是否要换


    # Evaluation主函数
    def run_eval(self):
        '''
        因子测试主函数
        :return:
        '''
        t = time.time()
        self.run_ic_test()
        print("IC Testing takes {}s".format(int(time.time()-t)))
        t = time.time()
        self.run_long_short_test() 
        print("Long Short takes {}s".format(int(time.time() - t)))
        t = time.time()
        self.get_group_res()
        print("Group Testing takes {}s".format(int(time.time() - t)))
        t = time.time()
        self.data_dict = {
            "factor_name": self.factor_name,
            "long_ret_series_no_cost": self.long_short_raw_res_dict["long_portfolio_ret_no_cost"],
            "short_ret_series_no_cost": self.long_short_raw_res_dict["short_portfolio_ret_no_cost"],
            "long_ret_series_after_cost": self.long_short_raw_res_dict["long_portfolio_ret_after_cost"],
            "short_ret_series_after_cost": self.long_short_raw_res_dict["short_portfolio_ret_after_cost"],
            "long_batch_avg_ret_list": self.long_batch_avg_ret_list,
            "short_batch_avg_ret_list": self.short_batch_avg_ret_list,
            "group_ret_series": self.group_ret_series_list,
            "index_ret_series": self.DataAssist.index_ret_series,
            "ic_list": self.ic_list,
            "rank_ic_list": self.rank_ic_list,
            "ic_stats_dict": self.ic_stats_dict,
            "rank_ic_stats_dict": self.rank_ic_stats_dict,
            "ic_decay": self.ic_decay,
            "rank_ic_decay": self.rank_ic_decay,
            "turnover": self.long_short_raw_res_dict["turnover_series"],
            "turnover_long": self.long_short_raw_res_dict["long_turnover_series"],
            "turnover_short": self.long_short_raw_res_dict["short_turnover_series"],
            "long_no_cost_summary": self.long_no_cost_summary,
            "short_no_cost_summary":self.short_no_cost_summary,
            "long_after_cost_summary":self.long_after_cost_summary,
            "short_after_cost_summary":self.short_after_cost_summary,
            "summary_no_cost": self.summary_no_cost,
            "summary_after_cost": self.summary_after_cost,
            "long_holding_stats_df": self.long_holding_stats_df,
            "short_holding_stats_df":self.short_holding_stats_df,
            "long_tov_holding_ratio":self.long_tov_holding_ratio,
            "short_tov_holding_ratio":self.short_tov_holding_ratio,
            "long_size_holding_ratio":self.long_size_holding_ratio,
            "short_size_holding_ratio":self.short_size_holding_ratio,
            "decay_result_dict":self.decay_result_dict,
            "decay_list":self.decay_list,
            "lead_result_dict":self.lead_result_dict,
            "lag_result_dict":self.lag_result_dict,
            "lead_lag_list":self.lead_lag_list,
            "split_ret_type_list":self.split_ret_type_list,
            "split_result_dict":self.split_result_dict,

            # riskplot 有关
            "long_batch_signal_df_list": self.long_batch_signal_df_list,
            "short_batch_signal_df_list": self.short_batch_signal_df_list
        }
        print("Data Saving takes {}s".format(int(time.time() - t)))




    def get_data_dict(self):

        self.data_dict = {

            "index_ret_series": self.DataAssist.index_ret_series,
            "ic_list": self.ic_list,
            "rank_ic_list": self.rank_ic_list,
            "ic_stats_dict": self.ic_stats_dict,
            "rank_ic_stats_dict": self.rank_ic_stats_dict,
            "ic_decay": self.ic_decay,
            "rank_ic_decay": self.rank_ic_decay,

        }
        return

    def run_ic_test(self, rank=False):
        """
        运行IC测试
        :return self.ic_stats:Dict
        """

        self.ic_list = self.calc_ic(self.factor_df, self.ret_n_df)
        self.ic_stats_dict = self.calc_ic_stats(sig_bar=0.02)
        self.ic_decay = self.calc_ic_decay(cfg.ic_decay)
        if rank:
            self.rank_ic_list = self.calc_rank_ic(self.factor_df, self.ret_df)
            self.rank_ic_stats_dict = self.calc_rank_ic_stats(sig_bar=0.02)
            self.rank_ic_decay = self.calc_ic_decay(cfg.ic_decay, rank=True)

    def run_long_short_test(self):
        """

        :return: 计算self.summary
        """
        # main test
        date_list = list(self.factor_df.columns)
        if self.freq == "W":
            batch_date_list = self.get_weekly_date_list(self.ret_df.T)
        else:
            batch_date_list = [date_list[i::self.adj_freq] for i in range(self.adj_freq)]

        self.long_short_raw_res_dict = self.get_long_short_res(batch_date_list)  # 计算多空分组情况，并确认因子是否为反向因子
        long_ret_series_no_cost = self.long_short_raw_res_dict["long_portfolio_ret_no_cost"]
        long_ret_series_after_cost = self.long_short_raw_res_dict['long_portfolio_ret_after_cost']
        long_turnover_series = self.long_short_raw_res_dict["long_turnover_series"]
        long_cost_series = self.long_short_raw_res_dict["long_cost_series"]
        short_ret_series_no_cost = self.long_short_raw_res_dict['short_portfolio_ret_no_cost']
        short_ret_series_after_cost = self.long_short_raw_res_dict['short_portfolio_ret_after_cost']
        short_turnover_series = self.long_short_raw_res_dict["short_turnover_series"]
        short_cost_series = self.long_short_raw_res_dict["short_cost_series"]
        self.long_batch_avg_ret_list = self.long_short_raw_res_dict["long_portfolio_ret_avg"]
        self.short_batch_avg_ret_list = self.long_short_raw_res_dict["short_portfolio_ret_avg"]
        index_ret_series = self.DataAssist.index_ret_series
        self.long_no_cost_summary = self.get_long_short_ret_stats(long_ret_series_no_cost, index_ret_series, long_turnover_series, long_cost_series)
        self.short_no_cost_summary = self.get_long_short_ret_stats(short_ret_series_no_cost, index_ret_series, short_turnover_series, short_cost_series)
        self.long_after_cost_summary = self.get_long_short_ret_stats(long_ret_series_after_cost, index_ret_series, long_turnover_series,long_cost_series)
        self.short_after_cost_summary = self.get_long_short_ret_stats(short_ret_series_after_cost, index_ret_series, short_turnover_series,short_cost_series)
        self.summary_no_cost = self.get_summary(self.long_no_cost_summary,self.short_no_cost_summary)
        self.summary_after_cost = self.get_summary(self.long_after_cost_summary,self.short_after_cost_summary)

        # lag / decay 测试
        self.decay_result_dict = {
            'long': [],
            'short': [],
            'long_short': []
        }

        self.lead_result_dict = {
            'long': [],
            'short': [],
            'long_short': []
        }

        self.lag_result_dict = {
            'long': [],
            'short': [],
            'long_short': []
        }

        # Decay测试
        for adj_freq in self.decay_list:
            batch_date_index_tmp = [list(range(i, len(date_list), adj_freq)) for i in range(adj_freq)]
            drop_index_list = [list(set(list(range(len(date_list)))) - set(batch_date_index_tmp[i])) for i in
                                     range(adj_freq)]
            ls_simple_dict = self.get_ls_simple_ret(drop_index_list,adj_freq)
            self.decay_result_dict['long'].append(ls_simple_dict['long'])
            self.decay_result_dict['short'].append(ls_simple_dict['short'])
            self.decay_result_dict['long_short'].append(ls_simple_dict['long_short'])
            
        # Lead测试
        for lead in self.lead_lag_list[0]:
            batch_date_index_tmp = [list(range(i, len(date_list), self.adj_freq)) for i in range(self.adj_freq)]
            drop_index_list = [list(set(list(range(len(date_list)))) - set(batch_date_index_tmp[i])) for i in
                               range(self.adj_freq)]
            ls_simple_dict = self.get_ls_simple_ret(drop_index_list,self.adj_freq,lag=lead)
            self.lead_result_dict['long'].append(ls_simple_dict['long'])
            self.lead_result_dict['short'].append(ls_simple_dict['short'])
            self.lead_result_dict['long_short'].append(ls_simple_dict['long_short'])

        # Lag测试
        for lag in self.lead_lag_list[1]:
            batch_date_index_tmp = [list(range(i, len(date_list), self.adj_freq)) for i in range(self.adj_freq)]
            drop_index_list = [list(set(list(range(len(date_list)))) - set(batch_date_index_tmp[i])) for i in
                               range(self.adj_freq)]
            ls_simple_dict = self.get_ls_simple_ret(drop_index_list,self.adj_freq,lag=lag)
            self.lag_result_dict['long'].append(ls_simple_dict['long'])
            self.lag_result_dict['short'].append(ls_simple_dict['short'])
            self.lag_result_dict['long_short'].append(ls_simple_dict['long_short'])


        ## long_short_split_test
        self.split_ret_type_list = self.DataAssist.split_ret_type_list
        self.split_ret_dict = self.DataAssist.split_ret_dict
        self.split_result_dict = {
            'long':[],
            'short':[],
            'long_short':[]
        }

        for split_type in self.split_ret_type_list:
            ls_simple_dict = self.get_split_ret(split_type)
            self.split_result_dict['long'].append(ls_simple_dict['long'])
            self.split_result_dict['short'].append(ls_simple_dict['short'])
            self.split_result_dict['long_short'].append(ls_simple_dict['long_short'])


    def get_summary(self, long_summary, short_summary):
        '''
        确定哪个是更高的那个summary
        :param long_summary:
        :param short_summary:
        :return:
        '''
        if long_summary['AlphaRtn']['sum'] > short_summary['AlphaRtn']['sum']:
            return long_summary
        else:
            return short_summary

    def get_split_ret(self,split_type):
        '''
        获取return收益
        :param split_type:
        :return:
        '''
        date_list = [list(self.factor_df.columns)]
        split_ret_df = self.split_ret_dict[split_type]
        long_batch_signal_df_list, short_batch_signal_df_list = self.get_ls_batch_signal(date_list)
        long_signal_df = long_batch_signal_df_list[0]
        short_signal_df = short_batch_signal_df_list[0]
        long_portfolio_ret = self.get_return_series_no_cost(split_ret_df,long_signal_df)
        short_portfolio_ret = self.get_return_series_no_cost(split_ret_df,short_signal_df)
        long_short_portfolio_ret = long_portfolio_ret - short_portfolio_ret
        ls_simple_dict = {}
        ls_simple_dict['long'] = np.sum(long_portfolio_ret)
        ls_simple_dict['short'] = np.sum(short_portfolio_ret)
        ls_simple_dict['long_short'] = np.sum(long_short_portfolio_ret)
        return ls_simple_dict   

    def get_ls_simple_ret(self,batch_date_list,adj_freq,lag=0):
        '''
        简化版的多空测试，只得到最终结果，主要用于lead lag decay测试
        :param batch_data_list:
        :return:
        '''
        long_batch_signal_df_list,short_batch_signal_df_list = self.get_ls_batch_signal_acc(batch_date_list, lag)
        long_ret_series_list,short_ret_series_list = [],[]
        for signal_df in long_batch_signal_df_list:
            long_ret_series_list.append(self.get_return_series_no_cost(self.ret_df,signal_df))
        for signal_df in short_batch_signal_df_list:
            short_ret_series_list.append(self.get_return_series_no_cost(self.ret_df,signal_df))
        long_portfolio_ret = pd.concat(long_ret_series_list, axis=1).mean(axis=1)
        short_portfolio_ret = pd.concat(short_ret_series_list, axis=1).mean(axis=1)
        long_short_portfolio_ret = long_portfolio_ret - short_portfolio_ret
        ls_simple_dict = {}
        ls_simple_dict['long'] = np.nansum(long_portfolio_ret)
        ls_simple_dict['short'] = np.nansum(short_portfolio_ret)
        ls_simple_dict['long_short'] = np.nansum(long_short_portfolio_ret)

        return ls_simple_dict

    def get_ls_batch_signal_acc(self, drop_list,lag=0):
        '''
        加速
        :param drop_list:
        :return:
        '''

        # 如果是lag模式，因子要向前或者向后shift，得到绩效
        factor_df = self.factor_df.shift(lag, axis=1)

        def ffill(arr):
            '''
            实现arr的ffill功能
            :param arr:
            :return:
            '''
            mask = np.isnan(arr)
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            out = arr[np.arange(idx.shape[0])[:, None], idx]
            return out

        long_batch_signal_df_list = []
        short_batch_signal_df_list = []
        test_factor_arr = (factor_df * self.up_feasible_stock_df).values
        # test_factor_arr += 0.0000001 * np.random.random(factor_df.shape)
        test_factor_arr_rank = test_factor_arr.argsort(axis=0).argsort(axis=0).astype(float) + 1
        test_factor_arr_rank[pd.isna(test_factor_arr)] = np.nan
        rank_max = np.nanmax(test_factor_arr_rank, axis=0)
        long_thrd = (rank_max - self.head + 1e-4).reshape(1, -1)
        short_thrd = (np.ones(long_thrd.shape) * self.head - 1e-4).reshape(1, -1)
        long_signal_arr = 1 * (test_factor_arr_rank >= long_thrd)
        short_signal_arr = 1 * (test_factor_arr_rank <= short_thrd)

        # if swapping needed ...
        if self.need_swapping:
            long_signal_arr, short_signal_arr = short_signal_arr, long_signal_arr

        for i in range(len(drop_list)):
            long_signal = copy.deepcopy(long_signal_arr).astype('float32')
            short_signal = copy.deepcopy(short_signal_arr).astype('float32')
            long_signal[:, drop_list[i]] = np.nan
            short_signal[:, drop_list[i]] = np.nan
            long_signal = ffill(long_signal)
            short_signal = ffill(short_signal)
            test_long_signal_df = pd.DataFrame(long_signal, index=factor_df.index, columns=factor_df.columns)
            test_short_signal_df = pd.DataFrame(short_signal, index=factor_df.index, columns=factor_df.columns)
            long_batch_signal_df_list.append(test_long_signal_df)
            short_batch_signal_df_list.append(test_short_signal_df)

        return long_batch_signal_df_list, short_batch_signal_df_list

    def get_long_short_res(self, batch_date_list):
        """
        :param batch_date_list:
        :return:返回平均
        """
        self.long_batch_signal_df_list, self.short_batch_signal_df_list = self.get_ls_batch_signal(batch_date_list)
        long_ret_series_after_cost_list, short_ret_series_after_cost_list = [], []
        long_ret_series_no_cost_list, short_ret_series_no_cost_list = [], []
        long_cost_series_list, short_cost_series_list = [], []
        long_turnover_series_list, short_turnover_series_list = [], []
        long_holding_stats_df = pd.DataFrame()
        short_holding_stats_df = pd.DataFrame()
        holding_ratio_dict = {
            'long_tov':pd.Series(),
            'short_tov':pd.Series(),
            'long_size':pd.Series(),
            'short_size':pd.Series()
        }

        # long组计算
        for signal_df in self.long_batch_signal_df_list:
            if len(long_holding_stats_df.index) == 0:
                long_holding_stats_df = self.get_holding_stats(signal_df)
                holding_ratio_dict['long_tov'] = self.get_holding_tov_ratios(signal_df)
                holding_ratio_dict['long_size'] = self.get_holding_size_ratios(signal_df)
            else:
                long_holding_stats_df += self.get_holding_stats(signal_df)
                holding_ratio_dict['long_tov'] += self.get_holding_tov_ratios(signal_df)
                holding_ratio_dict['long_size'] += self.get_holding_size_ratios(signal_df)
            long_cost_df = self.get_cost_df(self.cost, signal_df)
            long_cost_series = self.get_cost_series(long_cost_df,signal_df)
            long_cost_series_list.append(long_cost_series)
            long_ret_series_after_cost = self.get_return_series_after_cost(self.ret_df, signal_df, long_cost_df)
            long_ret_series_no_cost = self.get_return_series_no_cost(self.ret_df, signal_df)
            long_ret_series_after_cost_list.append(long_ret_series_after_cost)
            long_ret_series_no_cost_list.append(long_ret_series_no_cost)
            long_turnover_series_list.append(self.get_turnover_series(signal_df))

        # short组计算
        for signal_df in self.short_batch_signal_df_list:
            if len(short_holding_stats_df.index) == 0:
                short_holding_stats_df = self.get_holding_stats(signal_df)
                holding_ratio_dict['short_tov'] = self.get_holding_tov_ratios(signal_df)
                holding_ratio_dict['short_size'] = self.get_holding_size_ratios(signal_df)
            else:
                short_holding_stats_df += self.get_holding_stats(signal_df)
                holding_ratio_dict['short_tov'] += self.get_holding_tov_ratios(signal_df)
                holding_ratio_dict['short_size'] += self.get_holding_size_ratios(signal_df)
            short_cost_df = self.get_cost_df(self.cost, signal_df)
            short_cost_series = self.get_cost_series(short_cost_df,signal_df)
            short_cost_series_list.append(short_cost_series)
            short_ret_series_after_cost = self.get_return_series_after_cost(self.ret_df, signal_df, short_cost_df)
            short_ret_series_no_cost = self.get_return_series_no_cost(self.ret_df, signal_df)
            short_ret_series_after_cost_list.append(short_ret_series_after_cost)
            short_ret_series_no_cost_list.append(short_ret_series_no_cost)
            short_turnover_series_list.append(self.get_turnover_series(signal_df))

        self.long_holding_stats_df = long_holding_stats_df/len(self.long_batch_signal_df_list)
        self.long_tov_holding_ratio = holding_ratio_dict['long_tov']/len(self.long_batch_signal_df_list)
        self.long_size_holding_ratio = holding_ratio_dict['long_size']/len(self.long_batch_signal_df_list)
        self.short_holding_stats_df = short_holding_stats_df / len(self.short_batch_signal_df_list)
        self.short_tov_holding_ratio = holding_ratio_dict['short_tov']/len(self.short_batch_signal_df_list)
        self.short_size_holding_ratio = holding_ratio_dict['short_size']/len(self.short_batch_signal_df_list)


        long_portfolio_ret_no_cost = pd.concat(long_ret_series_no_cost_list, axis=1).mean(axis=1)
        short_portfolio_ret_no_cost = pd.concat(short_ret_series_no_cost_list, axis=1).mean(axis=1)
        long_portfolio_ret_after_cost = pd.concat(long_ret_series_after_cost_list, axis=1).mean(axis=1)
        short_portfolio_ret_after_cost = pd.concat(short_ret_series_after_cost_list, axis=1).mean(axis=1)
        long_cost_series = pd.concat(long_cost_series_list, axis=1).mean(axis=1)
        short_cost_series = pd.concat(short_cost_series_list, axis=1).mean(axis=1)
        long_turnover_df = pd.concat(long_turnover_series_list, axis=1)
        short_turnover_df = pd.concat(short_turnover_series_list, axis=1)
        long_turnover_series = long_turnover_df.sum(axis=1) / (long_turnover_df.replace(0, np.nan).count(axis=1))
        short_turnover_series = short_turnover_df.sum(axis=1) / (short_turnover_df.replace(0, np.nan).count(axis=1))

        # 默认为long组是多头，short组是空头，但也有可能因子是反向的，故需要交换一堆东西
        # swap if necesssary (如果所谓多组小于空组)（并非所有都swap，只swap需要回传或者会挂到self下的attributes)
        if long_portfolio_ret_after_cost.sum() < short_portfolio_ret_after_cost.sum():
            self.need_swapping = True  # indicator，下次就直接用这个indicator去swap
            long_portfolio_ret_no_cost, short_portfolio_ret_no_cost = short_portfolio_ret_no_cost, long_portfolio_ret_no_cost
            long_portfolio_ret_after_cost, short_portfolio_ret_after_cost = short_portfolio_ret_after_cost, long_portfolio_ret_after_cost
            long_cost_series, short_cost_series = short_cost_series, long_cost_series
            long_turnover_series, short_turnover_series = short_turnover_series, long_turnover_series
            long_turnover_df, short_turnover_df = short_turnover_df, long_turnover_df  # looks like this is unnecessary to swap 
            self.long_holding_stats_df, self.short_holding_stats_df = self.short_holding_stats_df, self.long_holding_stats_df
            self.long_tov_holding_ratio, self.short_tov_holding_ratio = self.short_tov_holding_ratio, self.long_tov_holding_ratio
            self.long_size_holding_ratio, self.short_size_holding_ratio = self.short_size_holding_ratio, self.long_size_holding_ratio
            long_ret_series_no_cost_list, short_ret_series_no_cost_list = short_ret_series_no_cost_list, long_ret_series_no_cost_list
            self.long_batch_signal_df_list, self.short_batch_signal_df_list = self.short_batch_signal_df_list, self.long_batch_signal_df_list 

        long_short_raw_res_dict = {
            "long_portfolio_ret_no_cost": long_portfolio_ret_no_cost,
            "short_portfolio_ret_no_cost": short_portfolio_ret_no_cost,
            "long_portfolio_ret_after_cost": long_portfolio_ret_after_cost,
            "short_portfolio_ret_after_cost": short_portfolio_ret_after_cost,
            "long_portfolio_ret_avg": [x.mean() for x in long_ret_series_no_cost_list],
            "short_portfolio_ret_avg": [x.mean() for x in short_ret_series_no_cost_list],
            "long_cost_series": long_cost_series,
            "short_cost_series": short_cost_series,
            "long_turnover_series": long_turnover_series,
            "short_turnover_series": short_turnover_series,
            "turnover_series": (long_turnover_series + short_turnover_series) / 2,
            "long_ret_series_list": long_ret_series_no_cost_list,
            "short_ret_series_list": short_ret_series_no_cost_list,
        }

        return long_short_raw_res_dict


    def get_ls_batch_signal(self, batch_date_list):
        """
        对信号进行分组，并将所有组的信号合并成一个大的ls_batch_signal_df_list
        """
        long_batch_signal_df_list = []
        short_batch_signal_df_list = []
        test_factor_df = self.factor_df * self.up_feasible_stock_df
        test_factor_rank = (
                test_factor_df # + 0.0000001 * np.random.rand(test_factor_df.shape[0], test_factor_df.shape[1])
        ).rank(axis=0, method='first', na_option="keep")
        short_signal_df = 1 * (test_factor_rank <= self.head)
        long_signal_df = 1 * ((test_factor_rank - (test_factor_rank.max(axis=0) - self.head + 1)) >= 0)

        # if swapping needed ... 
        if self.need_swapping:  # 第一次是肯定不会swap的，因为是false；在算过一次之后就有可能被swap
            long_signal_df, short_signal_df = short_signal_df, long_signal_df  
        # 因为up_feasible_stock需要shift -1，两个signal最后一天一定是nan，需要mask一次
        long_signal_df.iloc[:, -1] = np.nan
        short_signal_df.iloc[:, -1] = np.nan 
        date_list = list(self.factor_df.columns)

        # 每条路径多空信号单独处理
        for i in range(len(batch_date_list)):
            keep_list = batch_date_list[i]
            drop_list = list(set(date_list) - set(keep_list))
            test_long_signal_df = copy.deepcopy(long_signal_df)
            test_short_signal_df = copy.deepcopy(short_signal_df)
            test_long_signal_df[drop_list] = np.nan
            test_short_signal_df[drop_list] = np.nan
            test_long_signal_df = test_long_signal_df.fillna(method="ffill", axis=1).fillna(0)
            test_short_signal_df = test_short_signal_df.fillna(method="ffill", axis=1).fillna(0)
            long_batch_signal_df_list.append(test_long_signal_df)
            short_batch_signal_df_list.append(test_short_signal_df)

        return long_batch_signal_df_list, short_batch_signal_df_list

    def get_long_short_ret_stats(self, ret_series, index_ret_series, turnover_series, cost_series):
        """
        计算多空回测结果
        """
        # ret_series = ret_series.fillna(0)
        index_ret_series = index_ret_series.fillna(0)
        turnover_series = turnover_series.fillna(0)
        ret_stats_df = pd.DataFrame()
        ret_stats = {
            "AlphaRtn": [],
            "AlphaSharpe": [],
            "AlphaDrawdown": [],
            "Beta": [],
            "Return": [],
            "Drawdown": [],
            "Turnover": [],
            "Cost": [],
        }

        if len(ret_series) != len(index_ret_series):
            print("The index (dates) of stock and index return series not match.")
            print("ret-len:{},index-ret-len:{}".format(len(ret_series), len(index_ret_series)))
            return ret_stats
        ret_series = ret_series
        date_list = list(ret_series.index)
        year_list = sorted(list(set(x[:4] for x in date_list)))
        for i in range(len(year_list)):
            begin = year_list[i] + "0101"
            end = year_list[i] + "1231"
            # 可能会因为转置问题报错
            year_ret = ret_series[(ret_series.index >= begin) * (ret_series.index <= end)]

            year_index_ret = index_ret_series[(index_ret_series.index >= begin) * (index_ret_series.index <= end)]
            year_turnover_series = turnover_series[(turnover_series.index >= begin) * (turnover_series.index <= end)]
            year_cost_series = cost_series[(cost_series.index >= begin) * (cost_series.index <= end)]
            ret_stats = {
                "AlphaRtn": self.get_alpha_return(year_ret, year_index_ret) * 100,
                "AlphaSharpe": self.get_alpha_sharpe(year_ret, year_index_ret)*np.sqrt(252),
                "AlphaDrawdown": self.get_max_drawdown(year_ret - year_index_ret) * 100,
                "Beta": self.get_beta(year_ret, year_index_ret),
                "Return": self.get_return(year_ret) * 100,
                "Drawdown": self.get_max_drawdown(year_ret) * 100,
                "Turnover": year_turnover_series.mean(),
                "Cost": self.get_cost(year_cost_series) * 100,
            }
            ret_stats_df = pd.concat([ret_stats_df, pd.DataFrame(ret_stats, index=[year_list[i]])])
        # 计算总绩效
        all_ret_stats = {
            "AlphaRtn": self.get_alpha_return(ret_series, index_ret_series) * 100,
            "AlphaSharpe": self.get_alpha_sharpe(ret_series, index_ret_series)*np.sqrt(252),
            "AlphaDrawdown": self.get_max_drawdown(ret_series - index_ret_series) * 100,
            "Beta": self.get_beta(ret_series, index_ret_series),
            "Return": self.get_return(ret_series) * 100,
            "Drawdown": self.get_max_drawdown(ret_series) * 100,
            "Turnover": turnover_series.mean(),
            "Cost": self.get_cost(cost_series) * 100,
        }
        ret_stats_df = pd.concat([ret_stats_df, pd.DataFrame(all_ret_stats, index=["sum"])])
        return ret_stats_df.to_dict()

    def get_holding_tov_ratios(self,signal_df):
        signal_df = signal_df/signal_df
        pool_df = pd.DataFrame(data=np.ones(signal_df.shape), index=signal_df.index, columns=signal_df.columns)
        pool_df = pool_df * self.mask  # 对应的票池（和有因子值的地方）
        TR = self.DataAssist.eod_data_dict["TurnoverRate"]
        holding_tov_10 = (signal_df * (TR.rolling(10,axis=1).mean())).mean()
        pool_tov_10 = (pool_df * (TR.rolling(10,axis=1).mean())).mean()
        return holding_tov_10/pool_tov_10

    def get_holding_size_ratios(self,signal_df):
        signal_df = signal_df/signal_df
        pool_df = pd.DataFrame(data=np.ones(signal_df.shape), index=signal_df.index, columns=signal_df.columns)
        pool_df = pool_df * self.mask  # 对应的票池（和有因子值的地方）
        FMV = self.DataAssist.eod_data_dict["FloatMarketValue"] / 1e8
        holding_size_10 = (signal_df * (FMV.rolling(10,axis=1).mean())).mean()
        pool_size_10 = (pool_df * (FMV.rolling(10,axis=1).mean())).mean()
        return holding_size_10/pool_size_10

    def get_holding_stats(self, signal_df):
        """
        对持仓股票进行一些统计，包括过去1/5/20天换手，过去5/20天波动率，流通市值分布，股价
        pool_stats则是对全票池进行统计，作为参考标准
        :param signal_df:
        :return: holding_stats_dict
        """
        signal_df = signal_df/signal_df
        pool_df = pd.DataFrame(data=np.ones(signal_df.shape), index=signal_df.index, columns=signal_df.columns)
        pool_df = pool_df * self.mask  # 对应的票池（和有因子值的地方）
        TR = self.DataAssist.eod_data_dict["TurnoverRate"]
        Rtn = (self.DataAssist.eod_data_dict["ClosePrice"] / self.DataAssist.eod_data_dict["PreClosePrice"])
        FMV = signal_df * self.DataAssist.eod_data_dict["FloatMarketValue"] / 1e8
        Price = signal_df * self.DataAssist.eod_data_dict["ClosePrice"]
        holding_stats_dict = {
            "DailyTurnoverRate_1": (signal_df * TR.rolling(1, axis=1).mean().shift(1, axis=1)).sum().sum() / signal_df.sum().sum(),
            "DailyTurnoverRate_5": (signal_df * TR.rolling(5, axis=1).mean().shift(1, axis=1)).sum().sum() / signal_df.sum().sum(),
            "DailyTurnoverRate_20": (signal_df * TR.rolling(20, axis=1).mean().shift(1, axis=1)).sum().sum() / signal_df.sum().sum(),
            "DailyRtnStd_5": (signal_df * Rtn.rolling(5, axis=1).std().shift(1, axis=1)).sum().sum() * 100 / signal_df.sum().sum(),
            "DailyRtnStd_20": (signal_df * Rtn.rolling(20, axis=1).std().shift(1, axis=1)).sum().sum() * 100  / signal_df.sum().sum(),
            "FloatMV_25%": FMV.quantile(0.25).mean(),
            "FloatMV_50%": FMV.quantile(0.5).mean(),
            "FloatMV_75%": FMV.quantile(0.75).mean(),
            "Price_25%": Price.quantile(0.25).mean(),
            "Price_50%": Price.quantile(0.5).mean(),
            "Price_75%": Price.quantile(0.75).mean(),
        }
        FMV_pool = pool_df * self.DataAssist.eod_data_dict["FloatMarketValue"] / 1e8
        Price_pool = pool_df * self.DataAssist.eod_data_dict["ClosePrice"]
        pool_stats_dict = {
            "DailyTurnoverRate_1": (pool_df * TR.rolling(1, axis=1).mean().shift(1, axis=1)).sum().sum() / pool_df.sum().sum(),
            "DailyTurnoverRate_5": (pool_df * TR.rolling(5, axis=1).mean().shift(1, axis=1)).sum().sum() / pool_df.sum().sum(),
            "DailyTurnoverRate_20": (pool_df * TR.rolling(20, axis=1).mean().shift(1, axis=1)).sum().sum() / pool_df.sum().sum(),
            "DailyRtnStd_5": (pool_df * Rtn.rolling(5, axis=1).std().shift(1, axis=1)).sum().sum() * 100  / pool_df.sum().sum(),
            "DailyRtnStd_20": (pool_df * Rtn.rolling(20, axis=1).std().shift(1, axis=1)).sum().sum() * 100  / pool_df.sum().sum(),
            "FloatMV_25%": FMV_pool.quantile(0.25).mean(),
            "FloatMV_50%": FMV_pool.quantile(0.5).mean(),
            "FloatMV_75%": FMV_pool.quantile(0.75).mean(),
            "Price_25%": Price_pool.quantile(0.25).mean(),
            "Price_50%": Price_pool.quantile(0.5).mean(),
            "Price_75%": Price_pool.quantile(0.75).mean(),
        }
        holding_stats_df = pd.concat([pd.Series(holding_stats_dict), pd.Series(pool_stats_dict)],axis=1)
        holding_stats_df.columns = ["HoldingsAvg", "PoolAvg"]
        return holding_stats_df.T

    def get_style_series(self,signal_df):
        '''

        :param signal_df:
        :return:
        '''
        return

    def get_group_batch_signal(self, batch_date_list):
        """
        对信号进行分组，并将所有组的信号合并成一个大的ls_batch_signal_df_list
        """
        group_batch_signal_df_list = []
        for i in range(self.group_num):
            group_batch_signal_df_list.append([])
        test_factor_df = self.factor_df * self.up_feasible_stock_df
        test_factor_rank = (
                test_factor_df # + 0.0000001 * np.random.rand(test_factor_df.shape[0], test_factor_df.shape[1])
        ).rank(axis=0, method = 'first', na_option="keep")
        date_list = list(self.factor_df.columns)


        # 划分小组
        m = test_factor_rank.max() / self.group_num
        group_df_list = []
        for i in range(self.group_num):
            low_bar = i * m
            high_bar = (i + 1) * m
            group_df_list.append(1 * (test_factor_rank >= low_bar) * (test_factor_rank <= high_bar))
        group_df_list = group_df_list if self.need_swapping else group_df_list[::-1]

        # 每条路径多空信号单独处理
        for i in range(len(batch_date_list)):
            keep_list = batch_date_list[i]
            drop_list = list(set(date_list) - set(keep_list))
            for j in range(self.group_num):
                test_signal_df = copy.deepcopy(group_df_list[j])
                test_signal_df[drop_list] = np.nan
                test_signal_df = test_signal_df.fillna(method="ffill", axis=1).fillna(0)
                group_batch_signal_df_list[j].append(test_signal_df)

        return group_batch_signal_df_list


    def get_group_batch_signal_acc(self,drop_list):
        '''
        numpy版分组测试，简单速度优化
        :param batch_date_list:
        :return:
        '''

        def ffill(arr):
            '''
            实现arr的ffill功能
            :param arr:
            :return:
            '''
            mask = np.isnan(arr)
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            out = arr[np.arange(idx.shape[0])[:, None], idx]
            return out

        group_batch_signal_df_list = []
        for i in range(self.group_num):
            group_batch_signal_df_list.append([])
        test_factor_arr = (self.factor_df * self.up_feasible_stock_df).values   # 记得分组也要加上涨停版限制
        # test_factor_arr += 0.0000001 * np.random.random(self.factor_df.shape)
        test_factor_arr_rank = test_factor_arr.argsort(axis=0).argsort(axis=0).astype(float) + 1
        test_factor_arr_rank[pd.isna(test_factor_arr)] = np.nan
        rank_max = np.nanmax(test_factor_arr_rank, axis=0)
        # 划分小组
        m = rank_max / self.group_num
        group_arr_list = []
        for i in range(self.group_num):
            low_bar = i*m
            high_bar = (i+1)*m
            signal_arr = 1*((test_factor_arr_rank>=low_bar)*(test_factor_arr_rank<=high_bar))
            group_arr_list.append(signal_arr)
        group_arr_list = group_arr_list if self.need_swapping else group_arr_list[::-1]


        # 每条路径多空信号单独处理
        for i in range(len(drop_list)):
            for j in range(self.group_num):
                test_signal_arr = copy.deepcopy(group_arr_list[j]).astype("float32")
                test_signal_arr[:,drop_list[i]] = np.nan
                test_signal_arr = ffill(test_signal_arr)
                test_signal_df = pd.DataFrame(test_signal_arr, index=self.factor_df.index, columns=self.factor_df.columns)
                group_batch_signal_df_list[j].append(test_signal_df)

        return group_batch_signal_df_list

    def get_group_res(self):
        """
        :param
        :return: 分组测试收益
        """
        date_list = list(self.factor_df.columns)
        # # 原版使用
        # if self.freq == "W":
        #     batch_date_list = self.get_weekly_date_list(self.ret_df.T)
        # else:
        #     batch_date_list = [date_list[i::self.adj_freq] for i in range(self.adj_freq)]

        batch_date_index_tmp = [list(range(i, len(date_list), self.adj_freq)) for i in range(self.adj_freq)]
        drop_index_list = [list(set(list(range(len(date_list)))) - set(batch_date_index_tmp[i])) for i in
                           range(self.adj_freq)]
        group_num = self.group_num
        group_batch_signal_df_list = self.get_group_batch_signal_acc(drop_index_list)
        group_cost_series_list = []
        group_ret_series_list = []
        group_turnover_series_list = []
        for _ in range(group_num):
            group_cost_series_list.append([])
            group_ret_series_list.append([])
            group_turnover_series_list.append([])

        for i in range(group_num):
            batch_signal_df_list = group_batch_signal_df_list[i]
            for signal_df in batch_signal_df_list:
                group_ret_series_list[i].append(self.get_return_series_no_cost(self.ret_df, signal_df))

            group_ret_series_list[i] = pd.concat(group_ret_series_list[i], axis=1).mean(axis=1)
        self.group_ret_series_list = group_ret_series_list

    # 关IC测试部分 (算IC的部分必须用转置了，因为要取每天的序列来算corr)
    @staticmethod
    def calc_ic(factor_df, ret_df):
        """
        :param factor_df:
        :param ret_df:
        :return: IC
        """
        factors = factor_df.values
        rets = ret_df.values
        ic_list = []
        for f, r in zip(factors.T, rets.T):
            temp = np.vstack((f, r))
            temp = temp[:, ~np.isnan(temp).any(axis=0)]
            ic_list.append(np.corrcoef(temp)[0][1])
        ic_list = pd.Series(ic_list, index=ret_df.columns)
        return ic_list

    @staticmethod
    def calc_rank_ic(factor_df, ret_df):
        """
        :param ret_df:
        :param factor_df:
        :return: rankIC
        """
        factors = factor_df.rank(ascending=False).values
        rets = ret_df.rank(ascending=False).values
        ic_list = []
        for f, r in zip(factors.T, rets.T):
            temp = np.vstack((f, r))
            temp = temp[:, ~np.isnan(temp).any(axis=0)]
            ic_list.append(np.corrcoef(temp)[0][1])
        ic_list = pd.Series(ic_list, index=ret_df.columns)
        return ic_list

    def calc_ic_stats(self, sig_bar=0.02):
        """
        :param sig_bar:
        :return: IC stats dictionary
        :keyword: [ "IC mean", "IC mean+", "IC mean-","IC_std", "IC std+","IC std-","IC pos ratio","IC neg ratio",
                "IC sig ratio", "IR", "IC num con pos mean", "IC num con neg mean"]
                *最长连续正/负IC时段数
        """
        ic = pd.Series(self.ic_list).dropna()
        ic_stats = {"IC mean": np.nanmean(ic),
                    "IC mean+": np.nanmean(ic[ic > 0]),
                    "IC mean-": np.nanmean(ic[ic < 0]),
                    "IC std": np.nanstd(ic),
                    "IC std+": np.nanstd(ic[ic > 0]),
                    "IC std-": np.nanstd(ic[ic < 0]),
                    "IC pos ratio": len(ic[ic > 0]) / len(ic),
                    "IC neg ratio": len(ic[ic < 0]) / len(ic),
                    "IC sig ratio": len(ic[ic.abs() > sig_bar]) / len(ic),
                    "IR": np.nanmean(ic) / np.nanstd(ic)
                    # "IC cum con pos mean": self.get_longest_pos(ic),
                    # "IC cum con neg mean": self.get_longest_pos(-ic)
                    }
        return ic_stats

    def calc_rank_ic_stats(self, sig_bar=0.02):
        '''
        RankIC指标
        :param sig_bar:
        :return:
        '''
        rank_ic = pd.Series(self.rank_ic_list).dropna()
        rank_ic_stats = {"Rank IC mean": np.nanmean(rank_ic),
                         "Rank IC mean+": np.nanmean(rank_ic[rank_ic > 0]),
                         "Rank IC mean-": np.nanmean(rank_ic[rank_ic < 0]),
                         "Rank IC std": np.nanstd(rank_ic),
                         "Rank IC std+": np.nanstd(rank_ic[rank_ic > 0]),
                         "Rank IC std-": np.nanstd(rank_ic[rank_ic < 0]),
                         "Rank IC pos ratio": len(rank_ic[rank_ic > 0]) / len(rank_ic),
                         "Rank IC neg ratio": len(rank_ic[rank_ic < 0]) / len(rank_ic),
                         "Rank IC sig ratio": len(rank_ic[rank_ic.abs() > sig_bar]) / len(rank_ic),
                         "Rank IR": np.nanmean(rank_ic) / np.nanstd(rank_ic)
                         # "Rank IC cum con pos mean": self.get_longest_pos(rank_ic),
                         # "Rank IC cum con neg mean": self.get_longest_pos(-rank_ic)
                         }
        return rank_ic_stats

    def calc_ic_decay(self, length,rank=False):
        """
        计算IC的衰减
        :param length:
        :return: ic衰减率，rank_ic衰减率
        """
        if not rank:
            ic_decay = []
            for i in range(length):
                ret_df_shift = self.ret_df.shift(- i -1, axis=1).rolling(i, axis=1).sum()  # 横轴为时间，要在时间维度上shift
                ic_list = self.calc_ic(self.factor_df, ret_df_shift)
                ic_decay.append(np.nanmean(ic_list))
            return ic_decay
        else:
            rank_ic_decay = []
            for i in range(length):
                ret_df_shift = self.ret_df.shift(-i, axis=1)  # 横轴为时间，要在时间维度上shift
                rank_ic_list = self.calc_rank_ic(self.factor_df, ret_df_shift)
                rank_ic_decay.append(np.nanmean(rank_ic_list))
            return rank_ic_decay

    ######################################
    # 统计类支持函数，目的文件内其它函数进行调用 #
    ######################################

    @staticmethod
    def get_longest_pos(ts):
        test = np.array(ts).copy()
        test[test > 0] = 1
        test[test < 0] = 0
        for i in range(1, len(test)):
            test[i] = (test[i - 1] + test[i]) * test[i]
        return int(max(pd.Series(test).fillna(0)))

    def get_return_series_after_cost(self, ret_df, signal_df, cost_df):
        """
        已有signal_df后，合成完整的收益率序列
        :param cost_df:
        :param ret_df:
        :param signal_df:
        :return:
        """
        signal_ret_df = ret_df * signal_df
        signal_ret_series = (signal_ret_df - cost_df).sum(axis=0) / signal_df.sum(axis=0)
        signal_ret_series = signal_ret_series.replace([np.infty, -np.infty], 0)
        return signal_ret_series

    def get_return_series_no_cost(self,ret_df,signal_df):
        """
        已有signal_df后，合成完整的收益率序列
        :param ret_df:
        :param signal_df:
        :return:
        """
        signal_ret_df = ret_df * signal_df
        signal_ret_series = signal_ret_df.sum(axis=0) / signal_df.sum(axis=0)
        signal_ret_series = signal_ret_series.replace([np.infty, -np.infty], 0)
        return signal_ret_series

    def get_turnover_series(self, signal_df):
        """
        已有signal_df后，获取turnover序列
        :param signal_df:
        :return:
        """
        weight_df_raw = signal_df / signal_df.sum()
        weight_df = (weight_df_raw + weight_df_raw * 0).fillna(0)  # change inf to nan, and fillna
        turnover = np.abs(weight_df - weight_df.shift(1,axis=1)).sum(axis=0)
        turnover_filled = (turnover + turnover * 0).fillna(0)
        return turnover_filled

    @staticmethod
    def get_cost_df(cost, signal_df):
        """
        获取cost DataFrame
        :param cost:
        :param signal_df:
        :return:
        """
        cost_df = np.abs(cost * (signal_df.shift(1,axis=1)-signal_df)) / 2
        return cost_df

    @staticmethod
    def get_cost_series(cost_df,signal_df):
        '''
        获取cost序列
        :param cost_df:
        :param signal_df:
        :return:
        '''
        cost_series = cost_df.sum(axis=0)/signal_df.sum(axis=0)
        return cost_series

    @staticmethod
    def get_alpha_return(ret_series, index_ret_series):
        """
        计算超额收益
        :param ret_series:
        :param index_ret_series:
        :return: 超额收益之和
        """
        valid_index_ret_series = index_ret_series * (0 * ret_series + 1)
        return (ret_series - valid_index_ret_series).fillna(0).sum()

    @staticmethod
    def get_alpha_sharpe(ret_series, index_ret_series):
        """
        计算超额收益的夏普(不考虑无风险收益）
        :param ret_series:
        :param index_ret_series:
        :return:
        """
        valid_index_ret_series = index_ret_series * (0 * ret_series + 1)
        return (ret_series - valid_index_ret_series).mean() / (ret_series - valid_index_ret_series).std()

    @staticmethod
    def get_max_drawdown(ret_series):
        """
        计算最大回撤
        :param ret_series:
        :return:
        """
        ret_cum = ret_series.fillna(0).cumsum()
        return min(ret_cum - ret_cum.expanding().max())

    @staticmethod
    def get_beta(ret_series, index_ret_series):
        """
        计算投资组合的beta系数
        :param ret_series:
        :param index_ret_series:
        :return:
        """
        valid_index_ret_series = (index_ret_series * (0 * ret_series + 1)).dropna()
        valid_ret_series = (ret_series*(0 * index_ret_series + 1)).dropna()
        return np.cov(valid_ret_series, valid_index_ret_series)[0, 1] / np.cov(valid_ret_series, valid_index_ret_series)[1, 1]

    @staticmethod
    def get_return(ret_series):
        """
        计算收益
        :param ret_series:
        :return:
        """
        return ret_series.fillna(0).sum()

    @staticmethod
    def get_cost(cost_series):
        """
        得到cost值
        :param cost_df:
        :return:
        """
        cost = cost_series.replace(np.infty,0).fillna(0).sum()
        return cost

    @staticmethod
    def get_mdd_period(ret_series):
        """
        计算最大回撤区间长度
        :return:
        """
        pass

    @staticmethod
    def get_weekly_date_list(eod_df):
        """
        整体思路：获取所有的工作日，非交易日的工作日只是占位，并标注这是星期几的位置，后续步骤用下一个交易日fill
                    ->剔除整周不交易的时间（不需要填充，直接用下一周）
                    ->周内并非全部缺失，为了对齐序列长度，用上一个交易日填充（在目前的列表里，用来填充的“上一个交易日”会出现多次）
                    ->用week标签拆分成五个序列，五个序列应该等长或最大长度差为1
        """
        # 得到所有business day
        start_date = eod_df.index[0]
        end_date = eod_df.index[-1]
        all_buss_day = list(pd.date_range(start_date, end_date, freq='B'))
        holidays = sorted(list(set([x.strftime('%Y%m%d') for x in all_buss_day]) - set(eod_df.index)))
        blank = pd.DataFrame(columns=eod_df.columns, index=holidays)
        df = pd.concat([eod_df, blank]).sort_index()

        # 加星期的标签
        years = pd.to_datetime(pd.Series(df.index)).dt.isocalendar().year
        weeks = pd.to_datetime(pd.Series(df.index)).dt.isocalendar().week + 100 * years
        weeks.index = df.index
        weekdays = pd.to_datetime(pd.Series(df.index)).dt.weekday + 1
        weekdays.index = df.index
        week_df = pd.concat([weeks, weekdays], axis=1)
        week_df.columns = ["week", "weekday"]
        tagged_df = pd.concat([week_df, (~df.isna()).sum(axis=1)], axis=1)
        tagged_df.columns = ["week", "weekday", "valid"]
        tagged_df["valid"] = (tagged_df.valid / tagged_df.valid).fillna(0)

        # 对所有股票剔除整周不交易的时间（一般是春节or十一）
        t = tagged_df.groupby("week").sum().valid
        no_trade_weeks = list(t[t == 0].index)
        for w in no_trade_weeks:
            tagged_df = tagged_df.drop(tagged_df[tagged_df.week == w].index)

        # 将填充的日期进行替换，并截取Mon...Fri的日期序列
        tagged_df = tagged_df.reset_index()
        tagged_df.loc[tagged_df.valid == 0, ["index"]] = np.nan
        tagged_df["index"] = list(tagged_df["index"].fillna(method='ffill'))
        date_list = []
        for i in range(1, 6):
            date_list.append(list(tagged_df[tagged_df.weekday == i]["index"]))
        return date_list

class SignalEvaluator:
    """
    SignalEvaluator，仅进行一种测试
    """
    def __init__(self, data_assist, signal_df,signal_name):
        self.DataAssist = data_assist
        self.signal_df = signal_df
        self.signal_name = signal_name
        self.ret_df = self.DataAssist.ret_df
        self.ret_n_df = self.DataAssist.ret_n_df
        self.signal_df = self.signal_df.loc[self.ret_df.index]
        self.cost = self.DataAssist.cost
        self.stock_pool = self.DataAssist.stock_pool
        self.up_feasible_stock_df = self.DataAssist.up_feasible_stock_df
        self.down_feasible_stock_df = self.DataAssist.down_feasible_stock_df
        self.head = cfg.head
        self.freq = cfg.freq  # "W"表示按周调仓，否则按日调仓
        self.decay_list = cfg.decay_list
        if self.freq == "W":
            self.adj_freq = 5
        else:
            self.adj_freq = cfg.adj_freq  # 按n日调仓模式下
        # self.by_days = cfg.by_days  # 按日调仓或每周周几调仓
        self.group_num = cfg.group_num

        # 未实际赋值
        self.data_dict = {}
        self.summary = {}
        self.group_ret_series_list = []
        self.long_short_raw_res_dict = {}
        self.long_batch_signal_df_list = []
        self.short_batch_signal_df_list = []
        self.long_batch_avg_ret_list = []
        self.short_batch_avg_ret_list = []
        self.long_short_batch_avg_ret_list = []
        self.group_raw_res_dict = {}
        self.holding_stats_df = pd.DataFrame()
        self.pool_stats_dict = {}

        self.long_only = False  # True if only long, false means also has short 

    def run_signal_eval(self):
        '''
        信号测试主函数
        :return:
        '''
        t = time.time()
        self.run_long_short_signal_test()
        print("Signal Testing takes {}s".format(int(time.time() - t)))
        self.data_dict = {
            "signal_name":self.signal_name,
            "index_ret_series": self.DataAssist.index_ret_series,
            "long_ret_series_no_cost": self.long_short_singal_raw_res_dict["long_ret_series_no_cost"],
            "short_ret_series_no_cost": self.long_short_singal_raw_res_dict["short_ret_series_no_cost"],
            "long_ret_series_after_cost": self.long_short_singal_raw_res_dict["long_ret_series_after_cost"],
            "short_ret_series_after_cost": self.long_short_singal_raw_res_dict["short_ret_series_after_cost"],
            "turnover": self.long_short_singal_raw_res_dict["turnover_series"],
            "turnover_long": self.long_short_singal_raw_res_dict["long_turnover_series"],
            "turnover_short": self.long_short_singal_raw_res_dict["short_turnover_series"],
            "long_no_cost_summary": self.long_no_cost_summary,
            "short_no_cost_summary": self.short_no_cost_summary,
            "long_after_cost_summary": self.long_after_cost_summary,
            "short_after_cost_summary": self.short_after_cost_summary,
            "summary_no_cost": self.summary_no_cost,
            "summary_after_cost": self.summary_after_cost
        }

    def run_long_short_signal_test(self):
        '''
        利用信号运行多空测试
        :return:
        '''

        self.long_short_singal_raw_res_dict = self.get_long_short_signal_res(self.signal_df)
        long_ret_series_no_cost = self.long_short_singal_raw_res_dict["long_ret_series_no_cost"]
        long_ret_series_after_cost = self.long_short_singal_raw_res_dict['long_ret_series_after_cost']
        long_turnover_series = self.long_short_singal_raw_res_dict["long_turnover_series"]
        long_cost_series = self.long_short_singal_raw_res_dict["long_cost_series"]
        short_ret_series_no_cost = self.long_short_singal_raw_res_dict['short_ret_series_no_cost']
        short_ret_series_after_cost = self.long_short_singal_raw_res_dict['short_ret_series_after_cost']
        short_turnover_series = self.long_short_singal_raw_res_dict["short_turnover_series"]
        short_cost_series = self.long_short_singal_raw_res_dict["short_cost_series"]
        index_ret_series = self.DataAssist.index_ret_series
        self.long_no_cost_summary = self.get_long_short_signal_ret_stats(long_ret_series_no_cost, index_ret_series, long_turnover_series, long_cost_series)
        self.short_no_cost_summary = self.get_long_short_signal_ret_stats(short_ret_series_no_cost, index_ret_series, short_turnover_series, short_cost_series)
        self.long_after_cost_summary = self.get_long_short_signal_ret_stats(long_ret_series_after_cost, index_ret_series, long_turnover_series,long_cost_series)
        self.short_after_cost_summary = self.get_long_short_signal_ret_stats(short_ret_series_after_cost, index_ret_series, short_turnover_series,short_cost_series)
        self.summary_no_cost = self.get_summary(self.long_no_cost_summary,self.short_no_cost_summary)
        self.summary_after_cost = self.get_summary(self.long_after_cost_summary,self.short_after_cost_summary)

    def get_summary(self, long_summary, short_summary):
        '''
        确定哪个是更高的那个summary
        :param long_summary:
        :param short_summary:
        :return:
        '''
        if long_summary['AlphaRtn']['sum'] > short_summary['AlphaRtn']['sum']:
            return long_summary
        # elif np.isnan(short_summary['AlphaRtn']['sum']):  # 如果是nan, 只有多头，那就返回多头指标
        #     return long_summary 
        elif self.long_only:  # if only long, then trivially return long series
            return long_summary
        else:
            return short_summary


    def get_long_short_signal_res(self, signal_df):
        '''
        运行多空因子测试
        :param signal_df:
        :return:
        '''

        # 信号处理
        signal_df = signal_df * self.up_feasible_stock_df
        long_signal_df = signal_df.clip(lower=0)
        short_signal_df = - signal_df.clip(upper=0)
        # 如果没有空头 (例如权重优化部分，全部赋成nan)
        if ((short_signal_df <= 1e-6)  | (short_signal_df.isna())).all().all():
            short_signal_df[:] = np.nan  # convert all to nan 
            self.long_only = True

        # 回传long，short signal去做风险归因
        self.long_signal_df = long_signal_df
        self.short_signal_df = short_signal_df
        

        ### 多头测试
        long_cost_df = self.get_cost_df(self.cost, long_signal_df)
        long_cost_series = self.get_cost_series(long_cost_df,long_signal_df)
        long_cost_series = long_cost_series.replace(np.infty,0)
        long_cost_series = long_cost_series.replace(-np.infty, 0)
        long_ret_series_after_cost = self.get_return_series_after_cost(self.ret_df, long_signal_df, long_cost_df)
        long_ret_series_no_cost = self.get_return_series_no_cost(self.ret_df, long_signal_df)
        long_turnover_series = self.get_turnover_series(long_signal_df)

        ### 空头测试
        short_cost_df = self.get_cost_df(self.cost, short_signal_df)
        short_cost_series = self.get_cost_series(short_cost_df,short_signal_df)
        short_cost_series = short_cost_series.replace(np.infty,0)
        short_cost_series = short_cost_series.replace(-np.infty, 0)
        short_ret_series_after_cost = self.get_return_series_after_cost(self.ret_df, short_signal_df, short_cost_df)
        short_ret_series_no_cost = self.get_return_series_no_cost(self.ret_df, short_signal_df)
        short_turnover_series = self.get_turnover_series(short_signal_df)

        ###
        long_short_singal_raw_res_dict = {
            'long_ret_series_no_cost':long_ret_series_no_cost,
            'short_ret_series_no_cost':short_ret_series_no_cost,
            'long_ret_series_after_cost':long_ret_series_after_cost,
            'short_ret_series_after_cost':short_ret_series_after_cost,
            'long_turnover_series':long_turnover_series,
            'short_turnover_series':short_turnover_series,
            'turnover_series':(long_turnover_series+short_turnover_series)/2,
            'long_cost_series':long_cost_series,
            'short_cost_series':short_cost_series
        }

        return long_short_singal_raw_res_dict

    def get_long_short_signal_ret_stats(self,ret_series,index_ret_series,turnover_series,cost_series):
        '''
        计算多空回测结果
        :param ret_series:
        :param index_ret_series:
        :param turnover_series:
        :param cost_series:
        :return:
        '''
        # ret_series = ret_series.fillna(0)
        index_ret_series = index_ret_series.fillna(0)
        turnover_series = turnover_series.fillna(0)
        ret_stats_df = pd.DataFrame()
        ret_stats = {
            "AlphaRtn": [],
            "AlphaSharpe": [],
            "AlphaDrawdown": [],
            "Beta": [],
            "Return": [],
            "Drawdown": [],
            "Turnover": [],
            "Cost": []
        }
        if len(ret_series) != len(index_ret_series):
            print("The index (dates) of stock and index return series does not match.")
            print("ret-len:{},index-ret-len:{}".format(len(ret_series), len(index_ret_series)))
            return ret_stats
        ret_series = ret_series
        date_list = list(ret_series.index)
        year_list = sorted(list(set(x[:4] for x in date_list)))
        for i in range(len(year_list)):
            begin = year_list[i] + "0101"
            end = year_list[i] + "1231"
            # 可能会因为转置问题报错
            year_ret = ret_series[(ret_series.index >= begin) * (ret_series.index <= end)]
            year_index_ret = index_ret_series[(index_ret_series.index >= begin) * (index_ret_series.index <= end)]
            year_turnover_series = turnover_series[(turnover_series.index >= begin) * (turnover_series.index <= end)]
            year_cost_series = cost_series[(cost_series.index >= begin) * (cost_series.index <= end)]
            ret_stats = {
                "AlphaRtn": self.get_alpha_return(year_ret, year_index_ret) * 100,
                "AlphaSharpe": self.get_alpha_sharpe(year_ret, year_index_ret)*np.sqrt(252),
                "AlphaDrawdown": self.get_max_drawdown(year_ret - year_index_ret) * 100,
                "Beta": self.get_beta(year_ret, year_index_ret),
                "Return": self.get_return(year_ret) * 100,
                "Drawdown": self.get_max_drawdown(year_ret) * 100,
                "Turnover": year_turnover_series.mean(),
                "Cost": self.get_cost(year_cost_series) * 100,
            }
            ret_stats_df = pd.concat([ret_stats_df, pd.DataFrame(ret_stats, index=[year_list[i]])])
        # 计算总绩效
        all_ret_stats = {
            "AlphaRtn": self.get_alpha_return(ret_series, index_ret_series) * 100,
            "AlphaSharpe": self.get_alpha_sharpe(ret_series, index_ret_series)*np.sqrt(252),
            "AlphaDrawdown": self.get_max_drawdown(ret_series - index_ret_series) * 100,
            "Beta": self.get_beta(ret_series, index_ret_series),
            "Return": self.get_return(ret_series) * 100,
            "Drawdown": self.get_max_drawdown(ret_series) * 100,
            "Turnover": turnover_series.mean(),
            "Cost": self.get_cost(cost_series) * 100,
        }
        ret_stats_df = pd.concat([ret_stats_df, pd.DataFrame(all_ret_stats, index=["sum"])])
        return ret_stats_df.to_dict()


    ######################################
    # 统计类支持函数，目的文件内其它函数进行调用 #
    ######################################

    @staticmethod
    def get_longest_pos(ts):
        test = np.array(ts).copy()
        test[test > 0] = 1
        test[test < 0] = 0
        for i in range(1, len(test)):
            test[i] = (test[i - 1] + test[i]) * test[i]
        return int(max(pd.Series(test).fillna(0)))

    def get_return_series_after_cost(self, ret_df, signal_df, cost_df):
        """
        已有signal_df后，合成完整的收益率序列
        :param cost_df:
        :param ret_df:
        :param signal_df:
        :return:
        """
        signal_ret_df = ret_df * signal_df
        # signal_ret_series = signal_ret_df.sum(axis=0) / signal_df.sum(axis=0) - cost_df.sum(axis=0) / signal_df.sum(axis=0)
        signal_ret_series = (signal_ret_df - cost_df).sum(axis=0) / signal_df.sum(axis=0)
        signal_ret_series = signal_ret_series.replace(-np.infty, 0)
        signal_ret_series = signal_ret_series.replace(np.infty, 0)
        return signal_ret_series

    def get_return_series_no_cost(self, ret_df, signal_df):
        """
        已有signal_df后，合成完整的收益率序列
        :param ret_df:
        :param signal_df:
        :return:
        """
        signal_ret_df = ret_df * signal_df
        signal_ret_series = signal_ret_df.sum(axis=0) / signal_df.sum(axis=0)
        return signal_ret_series.replace([np.infty,-np.infty], 0)

    def get_turnover_series(self, signal_df):
        """
        已有signal_df后，获取turnover序列
        :param signal_df:
        :return:
        """
        # weight_df = signal_df/(signal_df.sum(axis=0)).fillna(0).replace(np.inf, 0)
        weight_df = (signal_df + signal_df * 0) # .fillna(0)
        turnover = np.abs(weight_df - weight_df.shift(1,axis=1)).sum(axis=0)
        if (turnover <= 1e-6).all():  # 如果全是0，不要这个series了
            turnover[:] = np.nan
        return turnover # .fillna(0).replace(np.infty, 0)

    @staticmethod
    def get_cost_df(cost, signal_df):
        """
        获取cost序列
        :param cost:
        :param signal_df:
        :return:
        """
        cost_df = np.abs(cost * (signal_df.shift(1, axis=1) - signal_df)) / 2
        return cost_df

    @staticmethod
    def get_cost_series(cost_df,signal_df):
        '''
        获取cost序列
        :param cost_df:
        :param signal_df:
        :return:
        '''
        cost_series = cost_df.sum(axis=0)/signal_df.sum(axis=0)
        return cost_series

    @staticmethod
    def get_alpha_return(ret_series, index_ret_series):
        """
        计算超额收益
        :param ret_series:
        :param index_ret_series:
        :return: 超额收益之和
        """
        valid_index_ret_series = index_ret_series*(0*ret_series+1)
        return (ret_series - valid_index_ret_series).fillna(0).sum()

    @staticmethod
    def get_alpha_sharpe(ret_series, index_ret_series):
        """
        计算超额收益的夏普(不考虑无风险收益）
        :param ret_series:
        :param index_ret_series:
        :return:
        """
        valid_index_ret_series = index_ret_series * (0 * ret_series + 1)
        return (ret_series - valid_index_ret_series).mean() / (ret_series - valid_index_ret_series).std()

    @staticmethod
    def get_max_drawdown(ret_series):
        """
        计算最大回撤
        :param ret_series:
        :return:
        """
        ret_cum = ret_series.fillna(0).cumsum()
        return min(ret_cum - ret_cum.expanding().max())

    @staticmethod
    def get_beta(ret_series, index_ret_series):
        """
        计算投资组合的beta系数
        :param ret_series:
        :param index_ret_series:
        :return:
        """
        valid_index_ret_series = (index_ret_series * (0 * ret_series + 1)).dropna()
        valid_ret_series = (ret_series*(0 * index_ret_series + 1)).dropna()
        return np.cov(valid_ret_series, valid_index_ret_series)[0, 1] / np.cov(valid_ret_series, valid_index_ret_series)[1, 1]

    @staticmethod
    def get_return(ret_series):
        """
        计算收益
        :param ret_series:
        :return:
        """
        return ret_series.fillna(0).sum()

    @staticmethod
    def get_cost(cost_series):
        """
        得到cost值
        :param cost_df:
        :return:
        """
        cost = cost_series.replace(np.infty,0).fillna(0).sum()
        return cost
