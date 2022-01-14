import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append("..")
from configuration import config as cfg

class Recorder:
    def __init__(self, data_dict,factor_name):
        """
        为了记录结果
        """
        self.pool = '&'.join(cfg.index_list) + '_' + ('fmv' if cfg.weight_index_by_fmv else 'equal')
        if cfg.use_independent_return_benchmark:
            self.pool += '_' + cfg.return_benchmark_index

        self.data_dict = data_dict
        self.summary_result_list = [
            factor_name,
            time.strftime("%H:%M:%S", time.localtime()),
            ','.join(cfg.factor_config['winsorize_method']),
            ','.join(cfg.factor_config['standardize_method']),
            ','.join(cfg.factor_config['neutralize_method']),
            self.pool,
            cfg.head,
            cfg.adj_freq,
            cfg.start_date,
            cfg.end_date,
            self.data_dict['ic_stats_dict']['IC mean'],
            self.data_dict['ic_stats_dict']['IC pos ratio'],
            self.data_dict['ic_stats_dict']['IC neg ratio'],
            self.data_dict['ic_stats_dict']['IC sig ratio'],
            self.data_dict['ic_stats_dict']['IR'],
            self.data_dict['summary_after_cost']['AlphaRtn']['sum'],
            self.data_dict['summary_no_cost']['AlphaRtn']['sum'],
            self.data_dict['summary_after_cost']['AlphaSharpe']['sum'],
            self.data_dict['summary_no_cost']['AlphaSharpe']['sum'],
            self.data_dict['summary_after_cost']['AlphaDrawdown']['sum'],
            self.data_dict['summary_no_cost']['AlphaDrawdown']['sum'],
            self.data_dict['summary_after_cost']['Return']['sum'],
            self.data_dict['summary_no_cost']['Return']['sum'],
            self.data_dict['summary_after_cost']['Drawdown']['sum'],
            self.data_dict['summary_no_cost']['Drawdown']['sum'],
            self.data_dict['summary_after_cost']['Turnover']['sum'],
            self.data_dict['summary_after_cost']['Cost']['sum'],
            self.data_dict['long_after_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['long_no_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['long_after_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['long_no_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['long_after_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['long_no_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['long_after_cost_summary']['Return']['sum'],
            self.data_dict['long_no_cost_summary']['Return']['sum'],
            self.data_dict['long_after_cost_summary']['Drawdown']['sum'],
            self.data_dict['long_no_cost_summary']['Drawdown']['sum'],
            self.data_dict['long_after_cost_summary']['Turnover']['sum'],
            self.data_dict['long_after_cost_summary']['Cost']['sum'],
            self.data_dict['short_after_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['short_no_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['short_after_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['short_no_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['short_after_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['short_no_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['short_after_cost_summary']['Return']['sum'],
            self.data_dict['short_no_cost_summary']['Return']['sum'],
            self.data_dict['short_after_cost_summary']['Drawdown']['sum'],
            self.data_dict['short_no_cost_summary']['Drawdown']['sum'],
            self.data_dict['short_after_cost_summary']['Turnover']['sum'],
            self.data_dict['short_after_cost_summary']['Cost']['sum']
        ]

class SignalRecorder:
    def __init__(self, data_dict,factor_name):
        """
        为了记录结果
        """
        self.pool = '&'.join(cfg.index_list) + '_' + ('fmv' if cfg.weight_index_by_fmv else 'equal')
        if cfg.use_independent_return_benchmark:
            self.pool += '_' + cfg.return_benchmark_index

        self.data_dict = data_dict
        self.summary_result_list = [
            factor_name,
            time.strftime("%H:%M:%S", time.localtime()),
            self.pool,
            cfg.head,
            cfg.adj_freq,
            cfg.start_date,
            cfg.end_date,
            self.data_dict['summary_after_cost']['AlphaRtn']['sum'],
            self.data_dict['summary_no_cost']['AlphaRtn']['sum'],
            self.data_dict['summary_after_cost']['AlphaSharpe']['sum'],
            self.data_dict['summary_no_cost']['AlphaSharpe']['sum'],
            self.data_dict['summary_after_cost']['AlphaDrawdown']['sum'],
            self.data_dict['summary_no_cost']['AlphaDrawdown']['sum'],
            self.data_dict['summary_after_cost']['Return']['sum'],
            self.data_dict['summary_no_cost']['Return']['sum'],
            self.data_dict['summary_after_cost']['Drawdown']['sum'],
            self.data_dict['summary_no_cost']['Drawdown']['sum'],
            self.data_dict['summary_after_cost']['Turnover']['sum'],
            self.data_dict['summary_after_cost']['Cost']['sum'],
            self.data_dict['long_after_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['long_no_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['long_after_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['long_no_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['long_after_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['long_no_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['long_after_cost_summary']['Return']['sum'],
            self.data_dict['long_no_cost_summary']['Return']['sum'],
            self.data_dict['long_after_cost_summary']['Drawdown']['sum'],
            self.data_dict['long_no_cost_summary']['Drawdown']['sum'],
            self.data_dict['long_after_cost_summary']['Turnover']['sum'],
            self.data_dict['long_after_cost_summary']['Cost']['sum'],
            self.data_dict['short_after_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['short_no_cost_summary']['AlphaRtn']['sum'],
            self.data_dict['short_after_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['short_no_cost_summary']['AlphaSharpe']['sum'],
            self.data_dict['short_after_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['short_no_cost_summary']['AlphaDrawdown']['sum'],
            self.data_dict['short_after_cost_summary']['Return']['sum'],
            self.data_dict['short_no_cost_summary']['Return']['sum'],
            self.data_dict['short_after_cost_summary']['Drawdown']['sum'],
            self.data_dict['short_no_cost_summary']['Drawdown']['sum'],
            self.data_dict['short_after_cost_summary']['Turnover']['sum'],
            self.data_dict['short_after_cost_summary']['Cost']['sum']
        ]




