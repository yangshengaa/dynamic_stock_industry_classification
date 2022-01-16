""" 
backtest a single factor 
"""

# load packages 
import os 
import sys
import time
import datetime

from typing import List, Dict
import pandas as pd


# load files 
from src.backtest.configuration import config as cfg
from src.backtest.tools.datatools import DataAssist
from src.backtest.tools.factools import FactorTools
from src.backtest.tools.graph import Grapher
from src.backtest.tools.evaluation import Evaluator
from src.backtest.tools.record import Recorder
from src.backtest.tools.risk_factorplot import risk_plotter


class SingleFactorBacktest:
    """
    日频单因子测试框架
    """

    def __init__(self, offline=False):
        # index returns 
        self.index_list = cfg.index_list
        pool_type = " + ".join(self.index_list) + ', ' + ('fmv weighted' if cfg.weight_index_by_fmv else 'equally weighted')
        self.pool_type = pool_type

        # stock pol
        self.fix_stocks = cfg.fix_stocks
        self.fixed_stock_pool = cfg.fixed_stock_pool

        # other configs
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        self.cost = cfg.cost
        self.return_type = cfg.return_type
        self.offline = offline  # in batch_factor_test, turn on offline mode, where eod_data_dict is passed in as a parameter
        self.head = cfg.head
        self.factor_path = cfg.factor_path
        
        # dates
        self.curr_date = datetime.datetime.today().strftime("%Y%m%d")

        # prepare
        self.DataAssist = None
        self.stock_pool = []
        self.factor_name = ''
        self.factor_df = None
        self.Evaluator = None
        self.Grapher = None

    def get_data(self, eod_data_dict=None):
        """
        read historical data

        :param eod_data_dict:
        :return:
        """
        if self.offline:
            self.DataAssist = DataAssist(offline=self.offline, eod_data_dict=eod_data_dict)
        else:
            print('---------- Retrieve Data -----------')
            self.DataAssist = DataAssist(offline=self.offline)
        self.stock_pool = self.DataAssist.stock_pool
        self.DataAssist.get_return_data()
        self.myconnector = self.DataAssist.myconnector
        return self.DataAssist.eod_data_dict

    def get_data_multi(self, eod_data_dict=None):
        """
        read historical data (for batch_factor_test)
        
        :param eod_data_dict:
        :return:
        """
        if self.offline:
            self.DataAssist = DataAssist(offline=self.offline, eod_data_dict=eod_data_dict)
        else:
            print('---------- Retrieve Data -----------')
            self.DataAssist = DataAssist(offline=self.offline)
        self.stock_pool = self.DataAssist.stock_pool
        self.DataAssist.get_return_data()
        return self.DataAssist.eod_data_dict

    # def read_factor_data(self, feature_name, tickers, date_list):
    #     """
    #     load factors
    #     """
    #     # load raw factors
    #     path = self.factor_path
    #     feature_name = "eod_" + feature_name
    #     factor = self.myconnector.get_eod_feature(fields=feature_name,
    #                                             where=path,
    #                                             tickers=tickers,
    #                                             dates=date_list)
    #     factor_df = factor[feature_name].to_dataframe()

    #     # 动态交易票池（跟随benchmark的成分股）
    #     if not self.fix_stocks:
    #         # 读取指数掩码
    #         index_mask = DataAssist.get_index_mask(self.index_list)
    #         factor_df_masked = factor_df * index_mask
    #     # 静态交易票池
    #     else: 
    #         factor_df_masked = factor_df + (factor_df.loc[self.fixed_stock_pool] - factor_df.loc[self.fixed_stock_pool])
        
    #     # 票池信息
    #     print(f'测试票池为: {(self.pool_type)}')

    #     # 加掩码
        
    #     return factor_df_masked

    def factor_process(self, factor_df: pd.DataFrame, factor_config: Dict[str, List]) -> pd.DataFrame:
        """
        process raw factors

        :param factor_df:
        :param params: 
        :return:
        """
        factor_df = factor_df + factor_df * 0  # 将inf转换成nan
        factor_df = FactorTools(
            self.DataAssist.eod_data_dict,
            factor_df,
            winsorize_method=factor_config['winsorize_method'],
            standardize_method=factor_config['standardize_method'],
            neutralize_method=factor_config['neutralize_method']
        ).processor()
        return factor_df

    def save_long_signal(self, long_batch_signal_df_list, short_batch_signal_df_list):
        """
        save long/short stock holdings. If the trade every n days, then there will be n csv. 

        :param *_batch_signal_df_list: 
        """
        # create save folders
        d = cfg.signal_df_output_path
        pool_type = '&'.join(cfg.index_list) +  '_' + ('fmv' if cfg.weight_index_by_fmv else 'equal')
        factor_name_saved = '{}_{}_{}_{}'.format(self.factor_name, pool_type, self.head, cfg.adj_freq)
        self.factor_name_saved = factor_name_saved
        save_path = '{}/{}'.format(d, factor_name_saved)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        # save csvs 
        # long holdings
        long_signal_df_paths = []
        for idx, long_signal_df in enumerate(long_batch_signal_df_list):
            long_signal_df_path = save_path + '/' + 'long' + '_' + str(idx) # + '.csv'
            # long_signal_df.astype(int).to_csv(long_signal_df_path)
            long_signal_df.astype(int).reset_index().to_feather(long_signal_df_path)
            long_signal_df_paths.append(long_signal_df_path)
        self.long_signal_df_paths = long_signal_df_paths  # save to self, facilitate plotting

        # short holdings
        short_signal_df_paths = []
        for idx, short_signal_df in enumerate(short_batch_signal_df_list):
            short_signal_df_path = save_path + '/' + 'short' + '_' + str(idx) # + '.csv'
            # short_signal_df.astype(int).to_csv(short_signal_df_path)
            short_signal_df.astype(int).reset_index().to_feather(short_signal_df_path)
            short_signal_df_paths.append(short_signal_df_path)
        self.short_signal_df_paths = short_signal_df_paths  # save to self，facilitate plotting

    
    def signal_riskplot(self):
        """
        signal risk plots
        """
        riskplot_save_path = '{}/facTest_riskplot_{}_{}'.format(cfg.output_path, self.curr_date, cfg.test_name)
        
        # long holding stats
        risk_plotter().plot(
            fig_name=self.factor_name_saved,
            csv_paths=self.long_signal_df_paths, 
            saving_path=riskplot_save_path,
            is_long=True
        )

        # short holding stats
        risk_plotter().plot(
            fig_name=self.factor_name_saved,
            csv_paths=self.short_signal_df_paths,
            saving_path=riskplot_save_path,
            is_long=False
        )


    def run_ds_factor(self, factor_df, factor_name):
        """
        run factor backtest
        """
        # factor values preprocessing
        self.factor_name = factor_name
        # self.factor_df = self.factor_process(factor_df,cfg.factor_config)
        self.factor_df = factor_df + factor_df * 0  # erase inf values
        # !!! just for barra 
        # self.factor_df = self.factor_df.shift(-1, axis=1) 
        # print('---------- Computeing Factors {}-----------'.format(self.factor_name))
        # t = time.time()
        # print('----------{} Computing Factors take {}s-----------\n'.format(self.factor_name, int(time.time() - t)))

        # IC，long/short，group test
        print('----------Start Testing {} -----------'.format(self.factor_name))
        t = time.time()
        self.Evaluator = Evaluator(self.DataAssist, self.factor_df, self.factor_name)
        self.Evaluator.run_eval()
        print('----------{} Testing takes {}s -----------\n'.format(self.factor_name, int(time.time() - t)))

        # graphing and reports
        print('----------Start Plotting {} -----------'.format(self.factor_name))
        t = time.time()
        self.Grapher = Grapher(self.Evaluator.data_dict)
        self.Grapher.save_fig(self.factor_name)
        self.Recorder = Recorder(self.Evaluator.data_dict, self.factor_name)
        # holding analysis
        self.save_long_signal(
            self.Evaluator.data_dict['long_batch_signal_df_list'], 
            self.Evaluator.data_dict['short_batch_signal_df_list']
        ) 

        if cfg.risk_plot_required:
            self.signal_riskplot()  # barra风险归因  # TODO: 修复Barra归因
        print('----------{} Plotting Takes{}s -----------\n'.format(self.factor_name, int(time.time() - t)))
        print('----------Testing Finishes {} -----------'.format(self.factor_name))
        return self.Recorder.summary_result_list


# if __name__ == '__main__':

#     t0 = time.time()
#     backtester = SingleFactorBacktest()
#     data = backtester.get_data()
#     print('----------数据读取用时{}s-----------\n'.format(int(time.time() - t0)))
#     factor_name = cfg.factor_name
#     date_list = list(data['ClosePrice'].columns)
#     stock_pool = backtester.stock_pool
#     factor_df = backtester.read_factor_data(factor_name, stock_pool, date_list)
#     result_list = backtester.run_ds_factor(factor_df, factor_name)
#     ## 存储数据
#     def save_record(summary_df):
#         d = cfg.output_path
#         curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
#         # 创建当日文件夹
#         save_path = '{}/facTest_file_{}_{}'.format(d, curr_date, cfg.test_name)
#         summary_file = '{}/summary.csv'.format(save_path)
#         try:
#             final_summary_df = pd.read_csv(summary_file)
#             del final_summary_df['Unnamed: 0']
#             final_summary_df = pd.concat([final_summary_df, summary_df])
#             final_summary_df.to_csv(summary_file)
#         except FileNotFoundError:
#             summary_df.to_csv(summary_file)

#     summary_df = pd.DataFrame(np.array([result_list]), columns=cfg.summary_cols)
#     save_record(summary_df)
