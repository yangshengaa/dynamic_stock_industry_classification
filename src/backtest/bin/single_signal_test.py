"""
test a single signal 
"""

# laod packages
import os 
import time
import datetime

# load files
from src.backtest.configuration import config as cfg
from src.backtest.tools.datatools import DataAssist
from src.backtest.tools.graph import SignalGrapher
from src.backtest.tools.evaluation import SignalEvaluator
from src.backtest.tools.record import SignalRecorder
from src.backtest.tools.risk_factorplot import risk_plotter


class SingleSignalBacktest:
    """
    signal backtest
    """
    def __init__(self, offline=False):
        # index return 
        self.index_list = cfg.index_list
        pool_type = " + ".join(self.index_list) + ', ' + ('fmv weighted' if cfg.weight_index_by_fmv else 'equally weighted')
        self.pool_type = pool_type
        # trade pool 
        self.fix_stocks = cfg.fix_stocks
        self.fixed_stock_pool = cfg.fixed_stock_pool

        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        self.cost = cfg.cost
        self.return_type = cfg.return_type
        self.offline = offline  # in batch testing, eod_data_dict will be passed in 
        self.head = cfg.head
        self.factor_path = cfg.factor_path
        self.read_from_feather = cfg.read_from_feather

        # dates
        self.curr_date = datetime.datetime.today().strftime("%Y%m%d")

        # prepare
        self.DataAssist = None
        self.stock_pool = []
        self.signal_name = ''
        self.signal_df = None
        self.Evaluator = None
        self.Grapher = None

    def get_data(self, eod_data_dict=None):
        """
        read data
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
        load data 
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

    # def read_signal_data(self, signal_name, tickers, date_list):
    #     path = self.factor_path

    #     # 如果是csv的方式，直接读取csv作为signal_df
    #     if self.read_from_csv:
    #         signal_df = pd.read_csv(f'{path}/{signal_name}', index_col=0)
    #         signal_df.index = [str(x).zfill(6) for x in signal_df.index]
    #         template_to_align_index = self.DataAssist.eod_data_dict['ClosePrice']
    #         signal_df = signal_df + (template_to_align_index - template_to_align_index)  # 对齐当前index
    #         signal_df = signal_df[template_to_align_index.columns]  # 对齐columns

    #     # 如果是h5，直接读入h5
    #     else:
    #         signal_name = "eod_" + signal_name
    #         signal = self.myconnector.get_eod_feature(fields=signal_name,
    #                                                 where=path,
    #                                                 tickers=tickers,
    #                                                 dates=date_list)
    #         signal_df = signal[signal_name].to_dataframe()
        
    #     # 对于weight和等量的signal，需要scale
    #     signal_df = signal_df / signal_df.sum() 
    #     signal_df = signal_df + signal_df * 0   # 将inf改成nan

    #     # 动态交易票池（跟随benchmark的成分股）
    #     if not self.fix_stocks:
    #     # 读取指数掩码
    #         index_mask = DataAssist.get_index_mask(self.index_list)
    #         signal_df_masked = signal_df * index_mask
    #     # 静态交易票池
    #     else:
    #         signal_df_masked = signal_df + (signal_df.loc[self.fixed_stock_pool] - signal_df.loc[self.fixed_stock_pool])
        
    #     # 票池信息
    #     print(f'测试票池为: {self.pool_type}')

    #     return signal_df_masked
    
    def signal_riskplot(self):
        """
        risk attribution 
        """
        riskplot_save_path = '{}/signalTest_riskplot_{}_{}'.format(cfg.output_path, self.curr_date, cfg.test_name)
        self.signal_name_saved = '{}_{}_{}_{}'.format(self.signal_name, self.pool_type, self.head, cfg.adj_freq)
        # output long holdings stats
        risk_plotter().plot(
            fig_name=self.signal_name_saved,
            csv_paths=[self.long_signal_df], 
            saving_path=riskplot_save_path,
            is_long=True
        )
        # short holding stats
        risk_plotter().plot(
            fig_name=self.signal_name_saved,
            csv_paths=[self.short_signal_df], 
            saving_path=riskplot_save_path,
            is_long=False
        )
        

    def run_signal(self, signal_df, signal_name):
        """
        signal backtest 
        """
        # TODO: better naming 
        self.signal_name = signal_name if not self.read_from_feather else signal_name.replace('/', '_')
        self.signal_df = signal_df
        # print('----------Retrieve Factors {}-----------'.format(self.signal_name))
        # t = time.time()
        # print('----------{}计算因子用时{}s-----------\n'.format(self.signal_name, int(time.time() - t)))

        print('---------- Start Testing {} -----------'.format(self.signal_name))
        t = time.time()
        self.Evaluator = SignalEvaluator(self.DataAssist, self.signal_df,self.signal_name)
        self.Evaluator.run_signal_eval()
        # extract long, short signals
        self.long_signal_df = self.Evaluator.long_signal_df
        self.short_signal_df = self.Evaluator.short_signal_df
        print('----------{} Testing takes {}s -----------\n'.format(self.signal_name, int(time.time() - t)))
        print('---------- Start Plotting {} -----------'.format(self.signal_name))
        t = time.time()
        self.Grapher = SignalGrapher(self.Evaluator.data_dict)
        self.Grapher.save_fig(self.signal_name)
        # riskplot
        if cfg.risk_plot_required:
            self.signal_riskplot()  # barra risk attribution
        print('----------{} Plotting Takes {}s -----------\n'.format(self.signal_name, int(time.time() - t)))
        self.SignalRecorder = SignalRecorder(self.Evaluator.data_dict,self.signal_name)
        print('----------Testing Finishes {} -----------'.format(self.signal_name))
        return self.SignalRecorder.summary_result_list


# if __name__ == '__main__':

#     t0 = time.time()
#     backtester = SingleSignalBacktest()
#     data = backtester.get_data()
#     print('----------数据读取用时{}s-----------\n'.format(int(time.time() - t0)))
#     signal_name = cfg.signal_name
#     date_list = list(data['ClosePrice'].columns)
#     stock_pool = backtester.stock_pool
#     signal_df = backtester.read_signal_data(signal_name,stock_pool,date_list)
#     result_list = backtester.run_signal(signal_df, signal_name)

#     ### 存储数据
#     def save_record(summary_df):
#         d = cfg.output_path
#         curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
#         # 创建当日文件夹
#         save_path = '{}/signal_test_file_{}'.format(d, curr_date)
#         summary_file = '{}/summary.csv'.format(save_path)
#         try:
#             final_summary_df = pd.read_csv(summary_file)
#             del final_summary_df['Unnamed: 0']
#             final_summary_df = pd.concat([final_summary_df, summary_df])
#             final_summary_df.to_csv(summary_file)
#         except FileNotFoundError:
#             summary_df.to_csv(summary_file)

#     summary_df = pd.DataFrame(np.array([result_list]), columns=cfg.signal_summary_cols)
#     save_record(summary_df)
