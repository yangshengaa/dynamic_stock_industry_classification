"""
Plot risk attributions (separated from the RiskPlot project)
TODO: 清除与AlphaTest重叠的部分
"""

# load packages
import os
import time
import datetime
import getpass 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
# from read_pos import ReadPos

# load files
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
import src.backtest.configuration.config as cfg

USER = getpass.getuser()

# ploting config 
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11


# plt.rc("font", family='WenQuanYi Micro Hei')

# TODO: 从外部读取这些东东，不要在函数里面读取（方便batch的shm）
class FactorPlot:
    def __init__(self, offline=False, benchmark_index='000852', return_type='o2next_o'):
        self.lw = 2
        self.return_type_list = ['o2next_o', 'c2next_o', 'o2c']
        self.return_type = return_type
        if self.return_type not in self.return_type_list:
            raise Exception(
                "Only 'o2next_o', 'c2next_o', 'o2c' are supported return types, please check your spelling"
            )
        self.shift_step_dict = {'o2next_o': 1, 'c2next_o': 1, 'o2c': 0}
        # print('FactorPlot Initialized')

        # data path 
        self.data_path = cfg.risk_fac_data_path 

        self.weight = pd.DataFrame()
        self.offline = offline

        # prepare
        self.idx_weight_dict = {}  # index member stock weights
        self.class_factor_dict = {}  # aggregated risk factor class
        self.idx_data_dict = {}  # index data 
        self.eod_data_dict = {}  # stock data
        self.ind_df = pd.DataFrame()  # industry one-hot
        self.plain_df = pd.DataFrame()  # plain one-hot
        self.ind_name_cn_lst = []  # 行业名称 # TODO: this is not available

        # stock returns 
        self.Open = pd.DataFrame()
        self.Close = pd.DataFrame()
        self.ret_df_dict = {}
        self.ret_df = pd.DataFrame()

        # factor returns
        self.factor_return_df = pd.DataFrame()
        self.ind_ret_df = pd.DataFrame()
        self.plain_ret_df = pd.DataFrame()
        self.all_ret_df = pd.DataFrame()

        # holding weights / benchmark weight 
        self.bm_index = benchmark_index
        self.w = pd.DataFrame()
        self.w_eq = pd.DataFrame()
        self.w_bm = pd.DataFrame()

        self.idx_list_all = ['000016', '000300', '000905', '000852']  # 包括创业板，用于绘制指数收益率
        if self.bm_index not in self.idx_list_all:
            raise KeyError(
                "Please use codes: '000300', '000905', '000852', other expressions are not implemented"
            )
        self.idx_list = ['000016', '000300', '000905', '000852']  # 用于持仓权重分类
        # self.output_path = './output/'  # 画图输出路径 

        self.style_attribute = pd.DataFrame()
        self.style_exposure = pd.DataFrame()

        self.ind_attribute = pd.DataFrame()
        self.ind_exposure = pd.DataFrame()

        self.plain_attribute = pd.DataFrame()
        self.plain_exposure = pd.DataFrame()

        # plotting related config 
        self.save_fig = True  # True to save plots, False otherwise

        self.all_stocks = []
        self.start_date = ''
        self.end_date = ''
        self.ds_end_date = ''
        self.idx_start_date = ''
        self.today_flag = False
        self.step = 100   # 最终绘图时日前间隔（太小了x轴的字会非常挤）
        self.today = False  # 是否用到最近一天的数据
        self.trade_dates = []
        self.ind_name_matcher = {
            'ind_1': 'Transportation', 'ind_2': 'LeisureService', 'ind_3': 'Media', 'ind_4': 'Utility',
            'ind_5': 'Agriculture', 'ind_6': 'Chem', 'ind_7': 'Bio-Medical',
            'ind_8': 'Trading', 'ind_9': 'Defence', 'ind_10': 'HomeApp', 'ind_11': 'Constrct-Material',
            'ind_12': 'Constrct-Decorate', 'ind_13': 'RealEst', 'ind_14': 'Metals', 'ind_15': 'Machinery',
            'ind_16': 'Auto', 'ind_17': 'E-Components', 'ind_18': 'E-Equip', 'ind_19': 'Textile', 'ind_20': 'Mixed',
            'ind_21': 'Computer', 'ind_22': 'LightMnfac', 'ind_23': 'Commun', 'ind_24': 'Mining',
            'ind_25': 'Steel', 'ind_26': 'Banks', 'ind_27': 'OtherFinance', 'ind_28': 'FoodBvrg'
        }

        # dataserver 
        # self.myconnector = PqiDataSdk(user=USER, size=1, pool_type="mp", log=False, offline=True, str_map=False)
        self.myconnector = PqiDataSdkOffline()

    def prepare(self):
        """
        load data
        """
        # print('Loading Data...')
        # if not os.path.exists('./output/'):
        #     os.mkdir('./output/')
        # if not os.path.exists('./output/' + str(datetime.datetime.now().date())):
        #     os.mkdir('./output/' + str(datetime.datetime.now().date()))
        # self.output_path = self.output_path + str(datetime.datetime.now().date()) + '/'
        if self.offline:  # TODO: examine offline 
            self.read_local_data()
        else:
            self.read_ds_data()
        self.calculator()
        self.adj_weight()

    def calculator(self):
        self.cal_ret_df()
        self.get_plain_df()
        self.calc_idx_rtn()

    # def plot_all(self):
    #     print('Plotting')
    #     self.step = int(len(self.ret_df.columns) / 2) + 1
    #     fig = plt.figure(figsize=(16, 32))
    #     spec = gridspec.GridSpec(ncols=5, nrows=24, figure=fig)
    #     self.plot_index_return(fig, spec[0:3, :])
    #     self.plot_weight_in_index(fig, spec[4:7, :])
    #     self.plot_risk_ts(fig, spec[8:11, :], spec[12:15, :])
    #     self.plot_risk_factor(fig, spec[16:19, :])
    #     self.plot_plain_factor(fig, spec[20:23, 0:2])
    #     self.plot_ind_factor(fig, spec[20:23, 3:])
    #     fig.tight_layout()
    #     plt.subplots_adjust(wspace=0.1, hspace=0.3)
    #     if self.save_fig:
    #         plt.savefig(self.output_path + 'HoldingAnalysis.png')

    def read_local_data(self):
        path = './Data/eod_data_dict'
        for f in os.listdir(path):
            temp = pd.read_csv(path + '/' + f).set_index('InstrumentID')
            temp.index = [str(x).zfill(6) for x in temp.index]
            self.eod_data_dict[f[:-4]] = temp

        path = './Data/idx_data_dict'
        for f in os.listdir(path):
            temp = pd.read_csv(path + '/' + f).set_index('InstrumentID')
            temp.index = [str(x).zfill(6) for x in temp.index]
            self.idx_data_dict[f[:-4]] = temp

        path = './Data/class_factor_dict'
        for f in os.listdir(path):
            temp = pd.read_csv(path + '/' + f).set_index('Unnamed: 0')
            temp.index = [str(x).zfill(6) for x in temp.index]
            self.class_factor_dict[f[:-4]] = temp

        path = './Data/idx_weight_dict'
        for f in os.listdir(path):
            temp = pd.read_csv(path + '/' + f).set_index('InstrumentID')
            temp.index = [str(x).zfill(6) for x in temp.index]
            self.idx_weight_dict[f[:-4]] = temp

        self.ind_df = pd.read_csv('./Data/ind_df.csv').set_index('InstrumentID')
        self.ind_df.index = [str(x).zfill(6) for x in self.ind_df.index]

        self.ind_name_cn_lst = ['银行', '纺织服装', '机械设备', '通信', '计算机', '交通运输', '有色金属', '电子', '电气设备', '采掘',
                                '国防军工', '建筑材料', '传媒', '农林牧渔', '房地产', '轻工制造', '医药生物', '综合', '建筑装饰',
                                '公用事业', '汽车', '家用电器', '钢铁', '商业贸易', '食品饮料', '非银金融', '化工', '休闲服务']

        self.factor_return_df = pd.read_csv('./Data/factor_return_df.csv').set_index('Unnamed: 0')
        self.ind_ret_df = pd.read_csv('./Data/ind_ret_df.csv').set_index('Unnamed: 0')

    def read_ds_data(self):
        """
        read data from dataserver 
        """
        # ds 
        self.all_stocks = self.myconnector.get_ticker_list()

        # stock eod
        self.ds_end_date = self.myconnector.get_next_trade_date(date=self.end_date)
        if self.ds_end_date == '':
            self.today_flag = True
            self.ds_end_date = self.end_date
        self.trade_dates = self.myconnector.select_trade_dates(
            start_date=self.start_date, end_date=self.ds_end_date
        )
        self.idx_start_date = self.myconnector.get_prev_trade_date(date=self.start_date)
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers=self.all_stocks, 
            start_date=self.start_date,
            end_date=self.ds_end_date
        )
        
        # index eod 
        self.idx_data_dict = self.myconnector.get_eod_history(
            tickers=self.idx_list_all,
            start_date=self.idx_start_date,
            end_date=self.ds_end_date, 
            source='index'
        )
        
        # align index
        self.template_align_tickers_df = (
            self.eod_data_dict['OpenPrice'] - self.eod_data_dict['OpenPrice']
        ).fillna(0).loc[:, self.start_date:self.end_date]

        # aggregated style factors eod
        # risk_fac_list = list(self.myconnector.get_feature_table(where=self.data_path + '/class_factors/').index)
        risk_fac_list = os.listdir(os.path.join(self.data_path, 'class_factors'))
        risk_fac_list = [x[4:] for x in risk_fac_list]  # remove 'eod_'

        # TODO 为什么还要在刷一下trade dates
        self.trade_dates = self.myconnector.select_trade_dates(start_date=self.start_date, end_date=self.end_date)
        if not self.end_date in self.trade_dates:
            self.trade_dates.append(self.end_date)
        else:
            pass
        # risk_factors = self.myconnector.get_eod_feature(
        #     fields=risk_fac_list, 
        #     tickers=self.all_stocks,
        #     where=self.data_path + '/class_factors/',
        #     dates=self.trade_dates
        # ).to_dict()
        # for k in risk_factors.keys():
        #     self.class_factor_dict[k[4:]] = risk_factors[k].to_dataframe().shift(-1, axis=1).loc[:, self.start_date:self.end_date]  # shift回今日
        for risk_factor_name in risk_fac_list:
            risk_factor_raw_df = self.myconnector.read_eod_feature(
                risk_factor_name, des='risk_factor/class_factors', dates=self.trade_dates
            )
            risk_factor_df = risk_factor_raw_df.shift(-1, axis=1).loc[:, self.start_date:self.end_date]  # shift back to today
            self.class_factor_dict[risk_factor_name] = risk_factor_df
        
        
        # industry class
        self.get_ind_df()

        # read from file 
        # index weight 
        path = self.data_path + '/idx_weight_dict/'
        for f in os.listdir(path):
            temp = pd.read_csv(path + '/' + f, index_col=0).loc[:, self.start_date:self.end_date]
            temp.index = [str(x).zfill(6) for x in temp.index]
            temp = temp + self.template_align_tickers_df
            temp.fillna(0, inplace=True)
            self.idx_weight_dict[f[:-4]] = temp

        # print('read plain_df data')
        # !!! 常用来临时更改做全票等权撮合
        # self.w_bm = (self.eod_data_dict['OpenPrice'] / self.eod_data_dict['OpenPrice']).loc[:, self.start_date:self.end_date]  # 全票等权
        self.w_bm = self.idx_weight_dict[self.bm_index]
        self.w_bm = self.w_bm.loc[self.w.index, :]
        self.w_bm = self.w_bm / self.w_bm.sum()

        # factor returns 
        self.factor_return_df = pd.read_csv(
            self.data_path + "/return_dfs/factor_return_df_dict/" + self.return_type + ".csv", index_col=0
        ).loc[self.start_date:self.end_date]

        # industry returns
        self.ind_ret_df = pd.read_csv(
            self.data_path + "/return_dfs/ind_ret_df_dict/" + self.return_type + ".csv", index_col=0
        ).loc[self.start_date:self.end_date]

        # plain returns
        self.plain_ret_df = pd.read_csv(
            self.data_path + "/return_dfs/plain_ret_df_dict/" + self.return_type + ".csv", index_col=0
        ).loc[self.start_date:self.end_date]

        # risk atribution（需对齐index）
        self.ind_risk_ra_df = pd.read_csv(
            self.data_path + "/return_dfs/path_ind_risk_ra/" + self.return_type + ".csv", index_col=0
        ).loc[:, self.start_date:self.end_date]
        self.ind_risk_ra_df.index = [str(x).zfill(6) for x in self.ind_risk_ra_df.index]
        self.ind_risk_ra_df = self.ind_risk_ra_df + self.template_align_tickers_df
        self.ind_risk_ra_df.sort_index(inplace=True)  # 为了.values操作

        # plain attribution（需对齐index）
        self.plain_ra_df = pd.read_csv(
            self.data_path + "/return_dfs/path_plain_ra/" + self.return_type + ".csv", index_col=0
        ).loc[:, self.start_date:self.end_date]
        self.plain_ra_df.index = [str(x).zfill(6) for x in self.plain_ra_df.index]
        self.plain_ra_df = self.plain_ra_df + self.template_align_tickers_df
        self.plain_ra_df.sort_index(inplace=True)     # 为了.values操作


    def get_plain_df(self):
        """ 
        0: sz50   (上证50)
        1: hs300  (沪深300, excluding sz50）
        2: zz500  (中证500)
        3: zz1000 (中证1000)
        4: kcb    (科创板)
        5: cyb    (创业板)
        6: others (anything else)
        """
        plain_df = pd.read_csv(self.data_path + '/plain_df_indicator/plain_df.csv', index_col=0)

        # align index and columns
        plain_df_selected = plain_df.loc[:, self.start_date:self.end_date]  # align dates
        plain_df_selected.index = [str(x).zfill(6) for x in plain_df_selected.index]
        plain_df_selected = plain_df_selected + self.template_align_tickers_df
        plain_df_selected.sort_index(inplace=True)

        # fill nan
        plain_df_selected.fillna(6, inplace=True)
        plain_df_selected.loc[plain_df_selected.index.str[:3] == '688'] = 4  # kcb
        plain_df_selected.loc[plain_df_selected.index.str[:1] == '3'] = 5    # cyb

        self.plain_df = plain_df_selected


    def adj_weight(self):
        """
        process scenarios when dates are missing 
        """
        nan_date_list = list(set(self.ret_df.columns) - set(self.w.columns))
        # ??? forward fill?
        if len(nan_date_list) > 0:
            fill = self.w.iloc[:, list(self.ret_df.columns).index(nan_date_list[0]) - 2]
            for date in nan_date_list:
                self.w.loc[:, date] = fill

        self.w = (self.ret_df - self.ret_df).fillna(0) + self.w
        self.w = self.w.fillna(0)
        self.w = self.w / self.w.sum()
        self.w_bm = (self.ret_df - self.ret_df).fillna(0) + self.w_bm
        self.w_bm = self.w_bm.fillna(0)


    def get_ind_df(self):
        # get industry class
        # TODO: move to config 
        self.ind_name_en_list = ['Transportation', 'LeisureService', 'Media', 'Utility', 'Agriculture', 'Chem',
                                 'Bio-Medical', 'Trading', 'Defence', 'HomeApp', 'Constrct-Material',
                                 'Constrct-Decorate', 'RealEst', 'Metals', 'Machinery', 'Auto',
                                 'E-Components', 'E-Equip', 'Textile', 'Mixed', 'Computer', 'LightMnfac', 'Commun',
                                 'Mining', 'Steel', 'Banks', 'OtherFinance', 'FoodBvrg']
        self.ind_name_cn_list = ['交通运输', '休闲服务', '传媒', '公用事业', '农林牧渔', '化工', '医药生物', '商业贸易', '国防军工',
                                 '家用电器', '建筑材料', '建筑装饰', '房地产', '有色金属', '机械设备', '汽车', '电子', '电气设备',
                                 '纺织服装', '综合', '计算机', '轻工制造', '通信', '采掘', '钢铁', '银行', '非银金融', '食品饮料']
        self.index_code_to_name = {
            '801010': '农林牧渔',
            '801020': '采掘',
            '801030': '化工',
            '801040': '钢铁',
            '801050': '有色金属',
            '801080': '电子', 
            '801110': '家用电器',
            '801120': '食品饮料',
            '801130': '纺织服装',
            '801140': '轻工制造',
            '801150': '医药生物',
            '801160': '公用事业',
            '801170': '交通运输',
            '801180': '房地产', 
            '801200': '商业贸易',
            '801210': '休闲服务',
            '801230': '综合',
            '801710': '建筑材料', 
            '801720': '建筑装饰',
            '801730': '电气设备',
            '801740': '国防军工',
            '801750': '计算机',
            '801760': '传媒',
            '801770': '通信', 
            '801780': '银行',
            '801790': '非银金融',
            '801880': '汽车',
            '801890': '机械设备'
        }
        
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


    def cal_ret_df(self):
        self.Open = self.eod_data_dict["OpenPrice"] * self.eod_data_dict["AdjFactor"]
        self.Close = self.eod_data_dict["ClosePrice"] * self.eod_data_dict["AdjFactor"]
        if self.return_type == 'o2next_o':
            self.ret_df = self.Open / self.Open.shift(1, axis=1) - 1
        elif self.return_type == 'c2next_o':
            self.ret_df = self.Open / self.Close.shift(1, axis=1) - 1
        elif self.return_type == 'o2c':
            self.ret_df = self.Close / self.Open - 1
        self.ret_df = self.ret_df.shift(-self.shift_step_dict[self.return_type], axis=1)
        if not self.today_flag:  # 低频组回测的时候必须过这一步
            self.ret_df = self.ret_df.iloc[:, :-1]

        self.w_eq = self.ret_df / self.ret_df
        self.w_eq = self.w_eq / self.w_eq.sum()


    def get_raw_df(self, weight, methd, black_lst=[]):
        # df的格式 col是dates index是ticker 数据两种方式输入方式 单票当日收盘手数/当日收盘市值 分析的是当日的Open2NextOpen
        all_stocks = self.myconnector.get_ticker_list()
        start_date = str(list(weight.columns)[0])
        end_date = str(list(weight.columns)[-1])
        eod_data_dict = self.myconnector.get_eod_history(
            tickers=all_stocks, start_date=start_date, end_date=end_date
        )
        prc_df = eod_data_dict['ClosePrice'].copy()
        Vwap = eod_data_dict['ClosePrice'].copy()  # ??? what is this for ?
        w_eq = Vwap / Vwap
        # pos_dict = {}
        dates = weight.columns.tolist()
        for tk in black_lst:
            weight = weight.drop(weight[weight.index == tk].index)

        weight.index = [str(x).zfill(6) for x in weight.index]
        if methd == 'vol':
            w = (weight * prc_df).dropna(axis=0)[dates]
        else:
            w = weight[dates]
        w = w / w.sum()
        w = ((w_eq - w_eq).fillna(0) + w.fillna(0)).fillna(0)
        self.get_weight(w)

    def get_weight(self, weight):
        self.w = weight
        self.start_date = list(self.w.columns)[0]
        self.end_date = list(self.w.columns)[-1]

    def plot_index_return(self, fig, spec):
        if self.today_flag and self.return_type == 'c2next_o':
            raise KeyError("Cannot analysis {} return with today's weight".format(self.return_type))
        ax = fig.add_subplot(spec)
        ax.plot(self.idx_data_dict['ClosePrice'].T / self.idx_data_dict['ClosePrice'].iloc[:, 0] - 1)
        ax.axhline(0, color='k', linestyle=':')
        ax.set_xticks(self.idx_data_dict['ClosePrice'].columns[::self.step])
        ax.legend(list(self.idx_data_dict['ClosePrice'].index))
        ax.grid()
        ax.set_title('Index cum return' + ' ', fontsize=18)

    def plot_weight_in_index(self, fig, spec):
        ax = fig.add_subplot(spec)
        index_indicator_dict = {}
        # 标记每天的指数成分股
        for idx_ticker in self.idx_weight_dict.keys():
            index_indicator_dict[idx_ticker] = 1 * (self.idx_weight_dict[idx_ticker] != 0)

        # 标记创业板
        w_idx_cyb = (self.ret_df - self.ret_df).fillna(0)
        w_idx_kcb = (self.ret_df - self.ret_df).fillna(0)
        idx_list_cyb = []
        idx_list_kcb = []
        for ticker in w_idx_cyb.index:
            if ticker.startswith('3'):
                idx_list_cyb.append(ticker)
            if ticker.startswith('688'):
                idx_list_kcb.append(ticker)
        w_idx_cyb.loc[idx_list_cyb] = 1
        w_idx_kcb.loc[idx_list_kcb] = 1
        index_indicator_dict['CYB'] = w_idx_cyb
        index_indicator_dict['KCB'] = w_idx_kcb

        # 不是以上任何一个指数成分
        w_idx = (self.ret_df / self.ret_df).fillna(1)
        for idx in index_indicator_dict.keys():
            w_idx = w_idx - index_indicator_dict[idx]
        index_indicator_dict['others'] = w_idx - index_indicator_dict['CYB'] - index_indicator_dict['KCB']
        index_indicator_dict['others'] = 1 * (index_indicator_dict['others'] == 1)

        # 计算持仓
        hold_value = []
        legend_lst = []
        for idx in index_indicator_dict.keys():
            hold_value.append((index_indicator_dict[idx] * self.w).sum())
            pct_mean = round(np.mean((index_indicator_dict[idx] * self.w).sum()) * 100, 2)
            legend_lst.append(str(idx) + ':' + str(pct_mean) + '%')
        # 画图
        date = list(self.w.columns)
        # color_list = ['grey', 'red', 'gold', 'cyan', 'steelblue', 'lime', 'aquamarine']
        color_list = ['black', 'navy', 'darkgreen', 'gold', 'grey', 'maroon', 'aquamarine']
        i = 0
        btm = 0
        last_v = 0
        for v in hold_value:
            if i == 0:
                btm = np.zeros(len(v))
            else:
                btm += last_v
            last_v = v
            # ax.axhline(np.mean(v), color=color_list[i])
            ax.bar(date, v, 0.5, bottom=btm, color=color_list[i])
            ax.set_xticks(date[::self.step])
            i += 1
        ax.legend(legend_lst, loc='upper left')
        ax.set_title(
            'Holding weight in Indexes' + ' ' + self.return_type,
            fontsize=18
        )

    def calc_holding_alpha(self):
        tmp = self.ret_df.loc[:, list(self.w.columns)].T.fillna(0).values
        hd_ret = np.nansum((self.ret_df.loc[:, list(self.w.columns)].T.fillna(0).values.dot(self.w.fillna(0).values)))
        index_rtn = self.idx_ret_df.loc[self.bm_index, :]
        return np.nansum(hd_ret - index_rtn)

    def calc_idx_rtn(self):
        # self.idx_ret_df = pd.DataFrame()
        # if self.return_type == 'o2c':
        #     self.idx_ret_df = self.idx_data_dict['ClosePrice'] / self.idx_data_dict['OpenPrice'] - 1
        # elif self.return_type == 'c2next_o':
        #     self.idx_ret_df = self.idx_data_dict['OpenPrice'] / self.idx_data_dict['ClosePrice'].shift(1, axis=1) - 1
        # elif self.return_type == 'o2next_o':
        #     self.idx_ret_df = self.idx_data_dict['OpenPrice'] / self.idx_data_dict['OpenPrice'].shift(1, axis=1) - 1
        # self.idx_ret_df = self.idx_ret_df.loc[:,str([x for x in list(self.plain_ret_df.index)][0])]
        self.idx_ret_df = pd.DataFrame()
        if self.return_type == 'o2c':
            self.idx_ret_df = self.idx_data_dict['ClosePrice'] / self.idx_data_dict['OpenPrice'] - 1
        elif self.return_type == 'c2next_o':
            self.idx_ret_df = self.idx_data_dict['OpenPrice'].shift(-1, axis=1) / self.idx_data_dict['ClosePrice'] - 1
        elif self.return_type == 'o2next_o':
            self.idx_ret_df = self.idx_data_dict['OpenPrice'].shift(-1, axis=1) / self.idx_data_dict['OpenPrice'] - 1
        self.idx_ret_df = self.idx_ret_df.loc[:, [str(i) for i in list(self.plain_ret_df.index)]]

    def plot_weight_dist(self, fig, spec):
        # 只支持单日风格分析
        last_weight = self.w.loc[:, [self.end_date]]
        last_weight = last_weight.drop(last_weight[last_weight[self.end_date] == 0].index)
        last_weight['w'] = last_weight[self.end_date]
        last_weight['ret'] = last_weight['w'] * self.ret_df.loc[last_weight.index, self.end_date]
        last_weight.dropna(axis=0)
        last_weight = last_weight.sort_values(by='w')
        colors1 = '#00CED1'  # 点的颜色
        colors2 = '#DC143C'
        ax = fig.add_subplot(spec)
        x = list(range(len(last_weight)))
        # print(last_weight)
        lns1 = ax.scatter(x, last_weight.w, c=colors1, alpha=0.9, label='weight')
        ax_ = ax.twinx()
        lns2 = ax_.scatter(x, last_weight.ret, c=colors2, alpha=0.9, label='return contribution')

        lns = [lns1, lns2]
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

        ax.axhline(y=np.mean(last_weight.w), color=colors1, linestyle='-')
        ax_.axhline(y=np.mean(last_weight.ret), color=colors2, linestyle='-')
        ax.set_title(
            'Holding Position Weight and Ret Contribution' + ' ' + self.return_type, 
            fontsize=18
        )
        ax.set_ylabel('Weight', fontsize=18)
        ax_.set_ylabel('Ret', fontsize=18)

    def plot_weight_in_index_single_day(self, fig, spec):
        ax = fig.add_subplot(spec)
        # ax.axis('off')

        index_indicator_dict = {}
        # 标记每天的指数成分股
        for idx_ticker in self.idx_weight_dict.keys():
            index_indicator_dict[idx_ticker] = 1 * (self.idx_weight_dict[idx_ticker] != 0)

        # 标记创业板
        w_idx_cyb = (self.ret_df - self.ret_df).fillna(0)
        w_idx_kcb = (self.ret_df - self.ret_df).fillna(0)
        idx_list_cyb = []
        idx_list_kcb = []
        for ticker in w_idx_cyb.index:
            if ticker.startswith('3'):
                idx_list_cyb.append(ticker)
            if ticker.startswith('688'):
                idx_list_kcb.append(ticker)
        w_idx_cyb.loc[idx_list_cyb] = 1
        w_idx_kcb.loc[idx_list_kcb] = 1
        index_indicator_dict['CYB'] = w_idx_cyb
        index_indicator_dict['KCB'] = w_idx_kcb

        # 不是以上任何一个指数成分
        w_idx = (self.ret_df / self.ret_df).fillna(1)
        for idx in index_indicator_dict.keys():
            w_idx = w_idx - index_indicator_dict[idx]
        index_indicator_dict['others'] = w_idx - index_indicator_dict['CYB'] - index_indicator_dict['KCB']
        index_indicator_dict['others'] = 1 * (index_indicator_dict['others'] == 1)
        # 计算持仓
        hold_value = []
        for idx in index_indicator_dict.keys():
            hold_value.append(np.round(100.0 * (index_indicator_dict[idx] * self.w).sum().iloc[0], 2))
        hold_value_bm = []
        for idx in index_indicator_dict.keys():
            hold_value_bm.append(np.round(100.0 * (index_indicator_dict[idx] * self.w_bm).sum().iloc[0], 2))
        labels = list(index_indicator_dict.keys())

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width / 2, hold_value, width, label='holdings')
        rects2 = ax.bar(x + width / 2, hold_value_bm, width, label=self.bm_index)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Weight in Portfolio')
        ax.set_title(
            'Holding weight in Indexes' + ' ' + self.return_type,
            fontsize=18
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        # # 画图
        # table = ax.table(cellText=[hold_value], colLabels=list(index_indicator_dict.keys()), loc='center',
        #                  cellLoc='center', rowLoc='center')
        # table.auto_set_font_size(False)
        # table.set_fontsize(10)  # 字体大小
        # table.scale(1, 2)

    def risk_factor_csv(self):
        """
        记录风格因子暴露的分年情况
        """
        exposure_df = []
        attribute_df = []
        year_list = list(set([i[:4] for i in self.style_exposure.index]))
        for year in year_list:
            exposure_df.append(self.style_exposure[(self.style_exposure.index >= year + '0000')
                            & (self.style_exposure.index < str(int(year) + 1) + '0000')].mean().rename(year))
            attribute_df.append(self.style_attribute[(self.style_attribute.index >= year + '0000')
                            & (self.style_attribute.index < str(int(year) + 1) + '0000')].fillna(0).cumsum().iloc[-1].rename(year))
            
        exposure_df.append(self.style_exposure.mean().rename('sum'))
        attribute_df.append(self.style_attribute.fillna(0).cumsum().iloc[-1].rename('sum'))
        
        exposure_df = pd.concat(exposure_df, axis=1).T  
        exposure_df.columns = [x + 'exp' for x in exposure_df.columns]
        attribute_df = pd.concat(attribute_df, axis=1).T
        attribute_df.columns = [x + 'attr' for x in attribute_df.columns]
        agg_df = pd.concat([exposure_df, attribute_df], axis=1)
        self.style_summary_df = agg_df
    
    def calc_attribution(self):
        if self.today_flag and self.return_type == 'c2next_o':
            raise KeyError("Cannot analysis {} return with today's weight".format(self.return_type))
        weight_df = self.w

        # 计算风格因子归因
        bm_attribute = pd.DataFrame()
        bm_exposure = pd.DataFrame()
        for fac_name in self.class_factor_dict.keys():
            X = self.class_factor_dict[fac_name]
            w1 = self.w_bm
            r = self.factor_return_df[fac_name]
            bm_attribute[fac_name] = (X * w1).sum().values * r.values
            bm_exposure[fac_name] = (X * w1).sum()
        bm_attribute.index = self.ret_df.columns
        bm_exposure.index = self.ret_df.columns

        attribute = pd.DataFrame()
        exposure = pd.DataFrame()
        for fac_name in self.class_factor_dict.keys():
            X = self.class_factor_dict[fac_name]
            w0 = weight_df
            r = self.factor_return_df[fac_name]
            attribute[fac_name] = (X * w0).sum().values * r.values
            exposure[fac_name] = (X * w0).sum()
        attribute.index = self.ret_df.columns
        exposure.index = self.ret_df.columns
        exposure = exposure - bm_exposure
        self.style_attribute = attribute - bm_attribute
        self.style_exposure = exposure
        self.risk_factor_csv()  # 处理分年的绩效

        # 计算行业归因
        self.ind_exposure = self.ind_df.T.dot((self.w.fillna(0) - self.w_bm))
        self.ind_attribute = (self.ind_df.T.dot(
            (self.w.fillna(0) - self.w_bm)).values * self.ind_ret_df.T.values)

        self.ind_attribute = pd.DataFrame(self.ind_attribute, index=self.ind_exposure.index,
                                          columns=self.ind_exposure.columns)
        # 计算板块归因
        self.return_type_list = ['o2next_o', 'c2next_o', 'o2c']
        
        
        self.idx_ret_df = pd.DataFrame(self.idx_ret_df.values, index=self.idx_ret_df.index, columns = self.plain_ret_df.index.tolist())
        self.plain_ret_df = pd.concat([self.plain_ret_df.T,self.idx_ret_df],axis=0)
        # 动态板块暴露
        excess_weight = self.w.fillna(0) - self.w_bm
        plain_exposure = []
        for date in excess_weight.columns:
            dynamic_plain_df = pd.get_dummies(self.plain_df[date])[list(range(7))]  # 保证顺序
            plain_exposure_by_date = dynamic_plain_df.T.dot(excess_weight[date])
            plain_exposure.append(plain_exposure_by_date)
        plain_exposure = pd.concat(plain_exposure, axis=1)
        plain_exposure.index = ['000016', '000300', '000905', '000852', 'kcb', 'cyb', 'others']
        plain_exposure.columns = excess_weight.columns
        self.plain_exposure = plain_exposure
        # 动态板块收益
        self.plain_ret_df = self.plain_ret_df.loc[self.plain_exposure.index, :]
        self.plain_attribute = (self.plain_exposure.values * (self.plain_ret_df-self.idx_ret_df.loc[self.bm_index].values))
        
        # 计算全因素归因
        ind_risk_ra = np.nansum(self.ind_risk_ra_df.fillna(0).values.T.dot(w0.values)) -np.nansum(self.ind_risk_ra_df.fillna(0).values.T.dot(w1.values))
        plain_ra = np.nansum(self.plain_ra_df.fillna(0).values.T.dot(w0.values)) -np.nansum(self.plain_ra_df.fillna(0).values.T.dot(w1.values))
        fac_dict = {'style': {'exposure': self.style_exposure.T, 'attributes': self.style_attribute.T},
                    'plain': {'exposure': self.plain_exposure, 'attributes': self.plain_attribute},
                    'ind': {'exposure': self.ind_exposure, 'attributes': self.ind_attribute}}

        return ind_risk_ra, plain_ra, self.style_attribute.sum().sum(), self.ind_attribute.sum().sum(), self.plain_attribute.sum().sum(), self.calc_holding_alpha(), fac_dict

    def plot_risk_ts(self, fig, spec1, spec2, spec3):
        if self.today_flag and self.return_type == 'c2next_o':
            raise KeyError("Cannot analysis {} return with today's weight".format(self.return_type))
        ax_1 = fig.add_subplot(spec1)
        ax_2 = fig.add_subplot(spec2)
        ax_3 = fig.add_subplot(spec3)
        color_dict = {
            'Reversal': 'crimson',
            'Momentum': 'tan', 
            'WeightedMomentum': 'tab:cyan',
            'IndustryMomentum': 'darkorange',

            'Volatility': 'navy', 
            'Turnover': 'green', 
            'Size': 'gold', 
            'NLSize': 'grey',

            'alpha': 'tab:purple',
            'betas': 'tab:brown', 
            'err_std': 'black'
        }
        # 风格暴露曲线
        # plot 1
        select_factors1 = ['Reversal', 'Momentum', 'WeightedMomentum', 'IndustryMomentum']
        exposure_selected = self.style_exposure.loc[:, select_factors1]
        if not self.today:
            exposure_selected.plot(ax=ax_1, color=[color_dict[col] for col in select_factors1], lw=self.lw)
        else:
            o2o = exposure_selected.index.tolist()[:-2]
            o2c = exposure_selected.index.tolist()[-3:-1]
            for col, row in exposure_selected.iteritems():
                ax_1.plot(exposure_selected.loc[o2o, col], '-o', c=color_dict[col], label=col, lw=self.lw, markersize=0.2)
                ax_1.plot(exposure_selected.loc[o2c, col], '--', c=color_dict[col], lw=self.lw)
                ax_1.set_xticks(exposure_selected.index[::self.step])
                ax_1.tick_params(axis='x', labelsize=13)
        ax_1.hlines(0, exposure_selected.index[0], exposure_selected.index[-1], color='k',
                    linestyle=':')
        ax_1.legend()
        ax_1.set_title(
            "Risk Factor Exposure 1" + ' ' + self.return_type,
            fontsize=18
        )

        # plot 2
        select_factors2 = ['Volatility', 'Turnover', 'Size', 'NLSize']
        exposure_selected = self.style_exposure.loc[:, select_factors2]
        if not self.today:
            exposure_selected.plot(ax=ax_2, color=[color_dict[col] for col in select_factors2], lw=self.lw)
        else:
            o2o = exposure_selected.index.tolist()[:-2]
            o2c = exposure_selected.index.tolist()[-3:-1]
            for col, row in exposure_selected.iteritems():
                ax_2.plot(exposure_selected.loc[o2o, col], '-o', c=color_dict[col], label=col, lw=self.lw, markersize=0.2)
                ax_2.plot(exposure_selected.loc[o2c, col], '--', c=color_dict[col], lw=self.lw)
                ax_2.set_xticks(exposure_selected.index[::self.step])
                ax_2.tick_params(axis='x', labelsize=13)
        ax_2.hlines(0, exposure_selected.index[0], exposure_selected.index[-1], color='k',
                    linestyle=':')
        ax_2.legend()
        ax_2.set_title("Risk Factor Exposure 2" + ' ' + self.return_type, fontsize=18)

        # plot 3
        select_factors3 = ['alpha', 'betas', 'err_std']
        exposure_selected = self.style_exposure.loc[:, select_factors3]
        if not self.today:
            exposure_selected.plot(ax=ax_3, color=[color_dict[col] for col in select_factors3], lw=self.lw)
        else:
            o2o = exposure_selected.index.tolist()[:-2]
            o2c = exposure_selected.index.tolist()[-3:-1]
            for col, row in exposure_selected.iteritems():
                ax_3.plot(exposure_selected.loc[o2o, col], '-o', c=color_dict[col], label=col, lw=self.lw, markersize=0.2)
                ax_3.plot(exposure_selected.loc[o2c, col], '--', c=color_dict[col], lw=self.lw)
                ax_3.set_xticks(exposure_selected.index[::self.step])
                ax_3.tick_params(axis='x', labelsize=13)
        ax_3.hlines(0, exposure_selected.index[0], exposure_selected.index[-1], color='k',
                    linestyle=':')
        ax_3.legend()
        ax_3.set_title("Risk Factor Exposure 3" + ' ' + self.return_type, fontsize=18)


    def plot_attr_ts(self, fig, spec1, spec2, spec3):
        if self.today_flag and self.return_type == 'c2next_o':
            raise KeyError("Cannot analysis {} return with today's weight".format(self.return_type))
        ax_1 = fig.add_subplot(spec1)
        ax_2 = fig.add_subplot(spec2)
        ax_3 = fig.add_subplot(spec3)
        color_dict = {
            'Reversal': 'crimson',
            'Momentum': 'tan', 
            'WeightedMomentum': 'tab:cyan',
            'IndustryMomentum': 'darkorange',

            'Volatility': 'navy', 
            'Turnover': 'green', 
            'Size': 'gold', 
            'NLSize': 'grey',

            'alpha': 'tab:purple',
            'betas': 'tab:brown', 
            'err_std': 'black'
        }
        # 风格暴露曲线
        # plot 1
        select_factors1 = ['Reversal', 'Momentum', 'WeightedMomentum', 'IndustryMomentum']
        attr_selected = self.style_attribute.loc[:, select_factors1]
        if not self.today:
            attr_selected.cumsum().plot(ax=ax_1, color=[color_dict[col] for col in select_factors1], lw=self.lw)
        else:
            o2o = attr_selected.index.tolist()[:-2]
            o2c = attr_selected.index.tolist()[-3:-1]
            for col, row in attr_selected.iteritems():
                ax_1.plot(attr_selected.loc[o2o, col].cumsum(), '-o', c=color_dict[col], label=col, lw=self.lw, markersize=0.2)
                ax_1.plot(attr_selected.loc[o2c, col].cumsum(), '--', c=color_dict[col], lw=self.lw)
                ax_1.set_xticks(attr_selected.index[::self.step])
                ax_1.tick_params(axis='x', labelsize=13)
        ax_1.hlines(0, attr_selected.index[0], attr_selected.index[-1], color='k',
                    linestyle=':')
        ax_1.legend()
        ax_1.grid()
        ax_1.set_title(
            "Risk Factor Attribute 1" + ' ' + self.return_type,
            fontsize=18
        )

        # plot 2
        select_factors2 = ['Volatility', 'Turnover', 'Size', 'NLSize']
        attr_selected = self.style_attribute.loc[:, select_factors2]
        if not self.today:
            attr_selected.cumsum().plot(ax=ax_2, color=[color_dict[col] for col in select_factors2], lw=self.lw)
        else:
            o2o = attr_selected.index.tolist()[:-2]
            o2c = attr_selected.index.tolist()[-3:-1]
            for col, row in attr_selected.iteritems():
                ax_2.plot(attr_selected.loc[o2o, col].cumsum(), '-o', c=color_dict[col], label=col, lw=self.lw, markersize=0.2)
                ax_2.plot(attr_selected.loc[o2c, col].cumsum(), '--', c=color_dict[col], lw=self.lw)
                ax_2.set_xticks(attr_selected.index[::self.step])
                ax_2.tick_params(axis='x', labelsize=13)
        ax_2.hlines(0, attr_selected.index[0], attr_selected.index[-1], color='k',
                    linestyle=':')
        ax_2.legend()
        ax_2.grid()
        ax_2.set_title("Risk Factor Attribute 2" + ' ' + self.return_type, fontsize=18)

        # plot 3 
        select_factors3 = ['alpha', 'betas', 'err_std']
        attr_selected = self.style_attribute.loc[:, select_factors3]
        if not self.today:
            attr_selected.cumsum().plot(ax=ax_3, color=[color_dict[col] for col in select_factors3], lw=self.lw)
        else:
            o2o = attr_selected.index.tolist()[:-2]
            o2c = attr_selected.index.tolist()[-3:-1]
            for col, row in attr_selected.iteritems():
                ax_3.plot(attr_selected.loc[o2o, col].cumsum(), '-o', c=color_dict[col], label=col, lw=self.lw, markersize=0.2)
                ax_3.plot(attr_selected.loc[o2c, col].cumsum(), '--', c=color_dict[col], lw=self.lw)
                ax_3.set_xticks(attr_selected.index[::self.step])
                ax_3.tick_params(axis='x', labelsize=13)
        ax_3.hlines(0, attr_selected.index[0], attr_selected.index[-1], color='k',
                    linestyle=':')
        ax_3.legend()
        ax_3.grid()
        ax_3.set_title("Risk Factor Attribute 3" + ' ' + self.return_type, fontsize=18)


    def plot_risk_factor(self, fig, spec):
        if self.today_flag and self.return_type == 'c2next_o':
            raise KeyError("Cannot analysis {} return with today's weight".format(self.return_type))
        ax = fig.add_subplot(spec)
        risk_df = pd.DataFrame()
        risk_df['attribute'] = self.style_attribute.fillna(0).cumsum().iloc[-1]
        risk_df['exposure'] = self.style_exposure.mean()
        risk_df = risk_df.sort_values('attribute', ascending=False)
        labels = risk_df.index
        mul = 100 * np.max(np.abs(risk_df['attribute'])) / np.max(np.abs(risk_df['exposure']))
        exposure = list(np.round(risk_df.exposure, 5))
        attribute = list(np.round(risk_df.attribute, 5) * (100 / mul))
        ylim = np.max(np.abs(attribute)) * 1.1
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width / 2, exposure, width, label='exposure')
        rects2 = ax.bar(x + width / 2, attribute, width, label='return attribute(%)')

        ax.set_ylabel('Exposure', fontsize=18)
        ax.set_title(
            'Style Exposure and Return Attribution' + ' ' + self.return_type,
            fontsize=18
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13, rotation=30, ha='right')
        ax.legend()
        ax_2 = ax.twinx()
        ax_2.set_ylabel('Attribution', fontsize=18)
        ax_2.set_xticks(x)
        ax_2.set_xticklabels(labels, fontsize=13, rotation=30, ha='right')

        def autolabel(rects, multi=1):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(np.round(multi * height, 3)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, np.sign(height) * 15 - 10),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12)

        autolabel(rects1)
        autolabel(rects2, mul)
        ax.set_ylim(-ylim, ylim)
        ax_2.set_ylim(-ylim * mul, ylim * mul)
        ax.axhline(0, color='k', linestyle=':')


    def plot_ind_factor(self, fig, spec):
        if self.today_flag and self.return_type == 'c2next_o':
            raise KeyError("Cannot analysis {} return with today's weight".format(self.return_type))
        ax = fig.add_subplot(spec)
        risk_df = pd.DataFrame()
        risk_df['attribute'] = self.ind_attribute.fillna(0).cumsum(axis=1).iloc[:, -1]
        risk_df['exposure'] = self.ind_exposure.mean(axis=1).values
        risk_df.index = self.ind_name_en_list
        risk_df = risk_df.sort_values('attribute', ascending=False)
        head_num = 5
        head_list = list(range(head_num)) + list(range(-head_num, 0))
        risk_df = risk_df.iloc[head_list, :]
        labels = risk_df.index
        mul = 100 * np.max(np.abs(risk_df['attribute'])) / np.max(np.abs(risk_df['exposure']))

        exposure = list(np.round(risk_df.exposure, 5))
        attribute = list(np.round(risk_df.attribute, 5) * (100 / mul))

        ylim = np.max(np.abs(attribute)) * 1.1

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width / 2, exposure, width, label='exposure')
        rects2 = ax.bar(x + width / 2, attribute, width, label='return attribute(%)')

        ax.set_ylabel('Exposure', fontsize=18)
        ax.set_title(
            'Industry Exposure and Return Attribution' + ' ' + self.return_type,
            fontsize=18
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13, rotation=30, ha='right')
        ax.legend()
        ax_2 = ax.twinx()
        ax_2.set_ylabel('Attribution', fontsize=18)

        def autolabel(rects, multi=1):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(np.round(multi * height, 3)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, np.sign(height) * 15 - 10),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12)

        # autolabel(rects1)
        autolabel(rects2, mul)
        ax.set_ylim(-ylim, ylim)
        ax_2.set_ylim(-ylim * mul, ylim * mul)
        ax.axhline(0, color='k', linestyle=':')


    def plot_plain_factor(self, fig, spec):
        if self.today_flag and self.return_type == 'c2next_o':
            raise KeyError("Cannot analysis {} return with today's weight".format(self.return_type))
        ax = fig.add_subplot(spec)
        risk_df = pd.DataFrame()
        risk_df['attribute'] = self.plain_attribute.fillna(0).cumsum(axis=1).iloc[:, -1]
        risk_df['exposure'] = self.plain_exposure.mean(axis=1).values
        risk_df = risk_df.sort_values('attribute', ascending=False)
        # head_list = list(range(head_num)) + list(range(-head_num, 0))
        # risk_df = risk_df.iloc[head_list, :]
        labels = risk_df.index
        mul = 100 * np.max(np.abs(risk_df['attribute'])) / np.max(np.abs(risk_df['exposure']))

        exposure = list(np.round(risk_df.exposure, 5))
        attribute = list(np.round(risk_df.attribute, 5) * (100 / mul))

        ylim = np.max(np.abs(attribute)) * 1.1

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width / 2, exposure, width, label='exposure')
        rects2 = ax.bar(x + width / 2, attribute, width, label='return attribute(%)')

        def autolabel(rects, multi=1):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(np.round(multi * height, 3)),
                            xy=(rect.get_x() + rect.get_width() / 2, height / 1.5),
                            xytext=(0, np.sign(height) * 15 - 10),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12)

        autolabel(rects2, mul)

        ax.set_ylabel('Exposure', fontsize=18)
        ax.set_title(
            'Plain Exposure and Return Attribution' + ' ' + self.return_type,
            fontsize=18
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13, rotation=30, ha='right')
        ax.legend()
        ax_2 = ax.twinx()
        ax_2.set_ylabel('Attribution', fontsize=18)


class risk_plotter:
    def __init__(self):
        pass
        # TODO: 对标票池的修改，可以从dataassist直接传进来
    
    def plot(self, fig_name='risk', csv_paths=[f'/home/{USER}/00 Factors/signals/signal_0.csv'], saving_path='', is_long=True):
        """
        riskplot绘图
        :param fig_name: 存储的因子名
        :param csv_paths: 一个list，装有读取持仓信号的路径或者dataframe
        :param saving_path: 存储最终因子图的路径
        :param is_long: True if it is long holdings, False otherwise. （只和最后导出的图的命名有关）
        """
        # 创建画图对象
        f = plt.figure(figsize=(24, 60), dpi=150)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        spec = gridspec.GridSpec(ncols=7, nrows=11, figure=f)

        # 导入数据 & 处理数据
        # 将调仓周期内的所有路径的信号相加取平均
        df = 0 
        for csv_path in csv_paths:
            if type(csv_path) == str:  # factor test
                # df = df + pd.read_csv(csv_path, index_col=0)
                df = df + pd.read_feather(csv_path).set_index('index')
            else:                      # signal test (不存储中间的持仓，直接导出)
                df = df + csv_path
        df = df / len(csv_path)
        plotter = FactorPlot(
            offline=False, 
            return_type='o2next_o', 
            benchmark_index=cfg.risk_plot_benchmark_index
        )  # return_type是归因的时间段 分为'o2next_o', 'c2next_o', 'o2c'， benchmark_index是标的指数，多日图现在只支持'o2next_o'
        plotter.get_raw_df(df, 'val')  # 如果df传入的是手数选择 vol 传入的是市值选择val
        plotter.prepare()
        plotter.calc_attribution()

        # 导出分年因子暴露图
        style_exp_attr_summary = plotter.style_summary_df
        style_summary_path = 'out/res/style_summary/'
        if not os.path.exists(style_summary_path):
            os.mkdir(style_summary_path)
        style_exp_attr_summary.to_csv(style_summary_path + f'style_summary_{fig_name}.csv')

        # 画出选择的风格分析图
        counter = 0

        # 指数日间收益
        plotter.plot_index_return(f, spec[counter:counter + 1, :])
        counter += 1

        # 持仓权重
        plotter.plot_weight_in_index(f, spec[counter:counter + 1, :])
        counter += 1

        # 风格暴露时序
        plotter.plot_risk_ts(f, spec[counter:counter + 1, :], spec[counter + 1:counter + 2, :], spec[counter + 2:counter + 3, :])
        counter += 3

        # 收益归因时序
        plotter.plot_attr_ts(f, spec[counter:counter + 1, :], spec[counter + 1:counter + 2, :], spec[counter + 2:counter + 3, :])
        counter += 3

        # 风格暴露图(overall)
        plotter.plot_risk_factor(f, spec[counter:counter + 1, :])
        counter += 1

        # 板块暴露/超额归因
        plotter.plot_plain_factor(f, spec[counter:counter + 1, 0:3])

        # 行业暴露/超额归因
        plotter.plot_ind_factor(f, spec[counter:counter + 1, 4:8])
        counter += 1


        plt.suptitle(f'{fig_name}_{"long" if is_long else "short"}', fontsize=25)
        f.subplots_adjust(top=0.95) 
        # # 仓位持仓权重的从大到小分布图 以及对应收益的从大到小分布图
        # plotter.plot_weight_dist(f, spec[counter:counter + 1, :])
        # 导出图片
        # print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
        # saving_path = saving_path #  + '/' + fig_name + '/'
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
        save_path = f'{saving_path}/{fig_name}_{"long" if is_long else "short"}.jpg'
        plt.savefig(save_path, bbox_inches='tight')

# # for testing purposes 
# if __name__ == '__main__':
#     rp = risk_plotter()
#     rp.plot(fig_name='mom_test',
#             csv_paths=[
#                 '/home/shengy/low_fre_alpha_tool/signal_df/mom_5_zzall_fmv_200_1/long_0.csv'
#             ],                      
#             saving_path='../res',
#             is_long=True)
