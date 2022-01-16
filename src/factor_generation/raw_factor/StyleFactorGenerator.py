
"""
Barra 风格因子生成项目（从原RiskPlot中分离出来)
"""

# load files 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
from src.factor_generation.raw_factor import style_factor_config as cfg

# load packages
import os
import time
import datetime
import logging
import warnings
import getpass

logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm

USER = getpass.getuser()


class StyleFactorGenerator(object):

    # ======================
    # ------ init ----------
    # ======================

    def __init__(self):
        # path config 
        self.saving_path = cfg.saving_path
        self.index_weight_path = cfg.index_weight_path
        self.index_code_to_weight = cfg.index_code_to_weight 

        # time frame 
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date

        # stock and index config 
        self.myconnector = PqiDataSdkOffline()
        self.all_stocks = self.myconnector.get_ticker_list()
        self.idx_list = cfg.index_list

        # prepare
        self.class_factor_dict_adj = {}
        self.idx_weight_dict = {}
        self.idx_ret_df_dict = {}
        self.tk_idx_ret_dict = {}
        self.style_class_dict = {}  # 按类别索引的细分风格因子
        self.ind_df = pd.DataFrame()  # 行业0-1矩阵
        self.plain_df = pd.DataFrame()  # 板块0-1矩阵
        self.class_factor_dict = {}  # 大类风格因子
        self.factor_return_df_dict = {}  
        self.all_ind_risk_ra_df_dict = {}
        self.all_plain_ra_df_dict = {}
        self.ind_ret_df_dict = {}
        self.plain_ret_df_dict = {}

        # load data
        self.load_data()

        # compute returns 
        self.compute_returns()

        # different types of returns need different shift length
        self.shift_step_dict = {'o2next_o': 1, 'c2next_o': 1, 'o2c': 0}
        self.ret_df = self.ret_df_dict['o2next_o'].shift(1, axis=1)  # * select type of returns
        self.date_list = list(self.Open.columns)


    def load_data(self):
        """
        load index and stock data 
        """
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers=self.all_stocks, 
            start_date=self.start_date,
            end_date=self.end_date
        )

        self.idx_dict = self.myconnector.get_eod_history(
            tickers=self.idx_list, 
            start_date=self.start_date,
            end_date=self.end_date, 
            source='index'
        )

    def compute_returns(self):
        """
        compute differnt modes of index returns and stock returns
        - o2c: open to close
        - o2next_o: open to next open 
        - c2next_o: close to next open
        """
        self.idx_ret_df_dict['o2c'] = self.idx_dict['ClosePrice'] / self.idx_dict['OpenPrice'] - 1
        self.idx_ret_df_dict['o2next_o'] = self.idx_dict['OpenPrice'].shift(-1, axis=1) / self.idx_dict['OpenPrice'] - 1
        self.idx_ret_df_dict['c2next_o'] = self.idx_dict['OpenPrice'].shift(-1, axis=1) / self.idx_dict['ClosePrice'] - 1

        # stock returns 
        self.ret_df_dict = {}
        self.return_type_list = ['o2next_o', 'c2next_o', 'o2c']
        self.Open = self.eod_data_dict["OpenPrice"] * self.eod_data_dict["AdjFactor"]
        self.Close = self.eod_data_dict["ClosePrice"] * self.eod_data_dict["AdjFactor"]
        self.ret_df_dict['o2next_o'] = self.Open.shift(-1, axis=1) / self.Open - 1
        self.ret_df_dict['c2next_o'] = self.Open.shift(-1, axis=1) / self.Close - 1
        self.ret_df_dict['o2c'] = self.Close / self.Open - 1

    # =================================
    # ------ compute factors ----------
    # =================================

    def get_ind_date(self):
        """
        read fixed industry data (申万一级行业分类)
        """
        # get industry information 
        ind_members = self.myconnector.get_sw_members().drop_duplicates(subset=['con_code'])
        # parse into a dataframe
        self.ind_df = pd.DataFrame(
            index=self.ret_df.index, columns=sorted(list(set(ind_members["index_code"])))
        )
        for i in range(len(ind_members.index)):
            label = list(ind_members.iloc[i])
            self.ind_df.loc[label[0], label[1]] = 1
        self.ind_df = self.ind_df.fillna(0)
        self.ind_df.columns = ["ind_" + str(i + 1) for i in range(len(self.ind_df.columns))]

        # print industry information
        print("stocks with classified industry: {}/{}".format(len(self.ind_df[self.ind_df.sum(axis=1) == 1].index), len(self.ind_df.index)))


    def get_one_index_weight(self, index_code: str) -> pd.DataFrame:
        """
        get index member stock weight within specified time frame

        :param index_code: supporting 000016, 000300, 000905, 000852
        :return a dataframe with stock wieghts 
        """
        # specify path 
        index_path = os.path.join(self.index_weight_path, index_code)
        # retrieve 
        index_data = pd.read_feather(index_path).set_index("index")
        df_selected = index_data.loc[:, self.date_list]
        return df_selected
    

    def get_plain_data(self):
        """
        compute a dynamic index member stock information as a dataframe, 
        whose index is stock code and columns are dates. Values are given by 0-6
        0: sz50
        1: hs300
        2: zz500
        3: zz1000
        4: kcb  (科创板)
        5: cyb  (创业板)
        6: others
        In particular, if a stock is in both hs300 and kcb, 
        then classify it as kcb (kcb and cyb have top priority)
        """
        # get index stock weight 
        sz50_wgt = self.get_one_index_weight('000016')
        hs300_wgt = self.get_one_index_weight('000300')
        zz500_wgt = self.get_one_index_weight('000905')
        zz1000_wgt = self.get_one_index_weight('000852')

        # convert to indicators
        sz50_indicator = sz50_wgt / sz50_wgt * 0 
        hs300_indicator = hs300_wgt / hs300_wgt * 1
        zz500_indicator = zz500_wgt / zz500_wgt * 2
        zz1000_indicator = zz1000_wgt / zz1000_wgt * 3
        
        # get stocks beloning to no indices
        others_position = sz50_indicator.isna() & hs300_indicator.isna() & zz500_indicator.isna() & zz1000_indicator.isna()

        # note that hs300, zz500, zz1000 are mutually exclusive, so we may add together
        plain_df = hs300_indicator.fillna(0) + zz500_indicator.fillna(0) + zz1000_indicator.fillna(0)
        plain_df.iloc[:, :] = np.where(sz50_indicator.notna(), 0, plain_df)  # sz50
        plain_df.iloc[:, :] = np.where(others_position, 6, plain_df)         # others

        # impose kcb and cyb respectively
        plain_df.loc[plain_df.index.str[:3] == '688'] = 4  # kcb
        plain_df.loc[plain_df.index.str[:1] == '3'] = 5    # cyb

        # add to self 
        self.plain_df = plain_df

        # save
        plain_df_saving_path = os.path.join(self.saving_path, 'plain_df_indicator')
        if not os.path.exists(plain_df_saving_path):
            os.mkdir(plain_df_saving_path)
        self.plain_df.astype(int).to_csv(os.path.join(plain_df_saving_path, 'plain_df.csv'))
    

    def calc_style_fac(self):
        """
        compute style factors (subclasses, to be aggregated to major classes later)
        """
        # exponential kernel 
        def exp_kernel_to_apply(s, kernel=None):
            """
            exponential decay kernl kernel
            :param s: series (a time series)
            :parma kernel: a numpy array 
            """
            return s @ kernel / kernel.sum()
        
        close_price = self.eod_data_dict['ClosePrice'] * self.eod_data_dict['AdjFactor']  # close price 自动ffill, 所以无需手动填nan
        close_price_return = close_price / close_price.shift(1, axis=1) - 1 

        # reversal (same as mom, 反向因子)
        reversal_hallife = 5
        reversal_one_minus_alpha = np.exp(-np.log(2) / reversal_hallife)  # from pandas documentation 
        reversal_kernel = reversal_one_minus_alpha ** np.array(range(39, -1, -1))
        rev_40 = close_price_return.rolling(40, axis=1).apply(exp_kernel_to_apply, args=(reversal_kernel, ), raw=True)
        rev_40 = rev_40 + (self.eod_data_dict['OpenPrice'] - self.eod_data_dict['OpenPrice'])  # add back mask
        self.style_class_dict['Reversal'] = {'rev_40': rev_40}

        # Momentum (in fact this is ret factor) (反向因子)
        momentum_halflife = 126   # CNE6 
        momentum_one_minus_alpha = np.exp(-np.log(2) / momentum_halflife) 
        momentum_kernel = momentum_one_minus_alpha ** np.array(range(251, -1, -1))
        mom_252 = close_price_return.shift(
            21,                 # get rid of short term momentum
            axis=1          
        ).rolling(
            252,                # long term
            axis=1
        ).apply(
            exp_kernel_to_apply, 
            args=(momentum_kernel, ), 
            raw=True
        )
        mom_252 = mom_252 + (self.eod_data_dict['OpenPrice'] - self.eod_data_dict['OpenPrice'])  # add back mask
        self.style_class_dict["Momentum"] = {'mom_252': mom_252}
                                            
        # Volatility (high_low 和 std均为反向因子)
        highest_price = self.eod_data_dict['HighestPrice'] * self.eod_data_dict['AdjFactor']
        lowest_price = self.eod_data_dict['LowestPrice'] * self.eod_data_dict['AdjFactor']
        high_low_5 = highest_price.rolling(5, axis=1).max() / lowest_price.rolling(5, axis=1).min()
        high_low_10 = highest_price.rolling(10, axis=1).max() / lowest_price.rolling(10, axis=1).min()
        high_low_20 = highest_price.rolling(20, axis=1).max() / lowest_price.rolling(20, axis=1).min()
        high_low_40 = highest_price.rolling(40, axis=1).max() / lowest_price.rolling(40, axis=1).min()
        high_low_120 = highest_price.rolling(120, axis=1).max() / lowest_price.rolling(120, axis=1).min()
        high_low_252 = highest_price.rolling(252, axis=1).max() / lowest_price.rolling(252, axis=1).min()

        close_over_preclose = self.eod_data_dict['ClosePrice'] / self.eod_data_dict['PreClosePrice']
        close_over_preclose += (self.eod_data_dict['OpenPrice'] - self.eod_data_dict['OpenPrice'])
        std_5 = close_over_preclose.rolling(5, axis=1).std()
        std_10 = close_over_preclose.rolling(10, axis=1).std()
        std_20 = close_over_preclose.rolling(20, axis=1).std()
        std_40 = close_over_preclose.rolling(40, axis=1).std()
        std_120 = close_over_preclose.rolling(120, axis=1).std()
        std_252 = close_over_preclose.rolling(252, axis=1).std()

        self.style_class_dict["Volatility"] = {
            'high_low_5': high_low_5,
            'high_low_10': high_low_10,
            'high_low_20': high_low_20,
            'high_low_40': high_low_40,
            'high_low_120': high_low_120,
            'high_low_252': high_low_252,
            'std_5': std_5,
            'std_10': std_10,
            'std_20': std_20,
            'std_40': std_40,
            'std_120': std_120,
            'std_252': std_252
        }
          
        # Turnover (Turnover 均为反向因子)
        turnover = self.eod_data_dict['TurnoverRate']
        turnover_1 = turnover.rolling(1, axis=1).mean()
        turnover_5 = turnover.rolling(5, axis=1).mean()
        turnover_10 = turnover.rolling(10, axis=1).mean()
        turnover_20 = turnover.rolling(20, axis=1).mean()
        turnover_40 = turnover.rolling(40, axis=1).mean()
        turnover_120 = turnover.rolling(120, axis=1).mean()
        turnover_252 = turnover.rolling(252, axis=1).mean()
        self.style_class_dict["Turnover"] = {
            'turnover_1': turnover_1,
            'turnover_5': turnover_5,
            'turnover_10': turnover_10,
            'turnover_20': turnover_20,
            'turnover_40': turnover_40,
            'turnover_120': turnover_120,
            'turnover_252': turnover_252
        }

        # Weighted Momentum (weighted momentum 均为反向因子)
        w_mom_1 = (turnover * (close_over_preclose - 1)).rolling(1, axis=1).mean() / turnover_1
        w_mom_5 = (turnover * (close_over_preclose - 1)).rolling(5, axis=1).mean() / turnover_5
        w_mom_10 = (turnover * (close_over_preclose - 1)).rolling(10, axis=1).mean() / turnover_10
        w_mom_20 = (turnover * (close_over_preclose - 1)).rolling(20, axis=1).mean() / turnover_20
        w_mom_40 = (turnover * (close_over_preclose - 1)).rolling(40, axis=1).mean() / turnover_40
        w_mom_120 = (turnover * (close_over_preclose - 1)).rolling(120, axis=1).mean() / turnover_120
        w_mom_252 = (turnover * (close_over_preclose - 1)).rolling(252, axis=1).mean() / turnover_252
        self.style_class_dict["WeightedMomentum"] = {
            'w_mom_1': w_mom_1,
            'w_mom_5': w_mom_5,
            'w_mom_10': w_mom_10,
            'w_mom_20': w_mom_20,
            'w_mom_40': w_mom_40,
            'w_mom_120': w_mom_120,
            'w_mom_252': w_mom_252
        }

        # Size (size为反向因子)
        size = np.log10(self.eod_data_dict["TotalMarketValue"])
        self.style_class_dict["Size"] = {'size': size}

        # Industry Momentum (短周期的industry momentum为正向因子, 60及以上有反向的意思)
        ind_weight = (self.ind_df / self.ind_df.sum())
        ind_weight_adj = 1 - ind_weight.sum(axis=1)
        ind_ret = (ind_weight_adj * (self.ind_df.dot(ind_weight.T.dot(self.ret_df.fillna(0))).T - (ind_weight.sum(axis=1) * self.ret_df.fillna(0).T))).T

        params = [1, 5, 10, 20, 60, 120]
        ind_mom_dict = {}
        for t in params:
            ind_mom_dict["Ind_MOM_" + str(t)] = ind_ret.rolling(t, axis=1).mean() + (self.Open - self.Open)
        self.style_class_dict["IndustryMomentum"] = ind_mom_dict


# # TODO: add 财务因子
#     def calc_fund_style_fac(self):
#         """
#         计算基本面相关的财务因子值
#         """
#         # del self.style_class_dict["Leverage"]
#         # del self.style_class_dict["Quality"]
#         # del self.style_class_dict["Values"]
#         # del self.style_class_dict["Growth"]
#         # TODO: book to price 
        
#         # leverage                                
#         self.style_class_dict["Leverage"] = {
#             'MLEV': self.fund_data_dict['MARKET_LEVERAGE_TTM'],
#             'BLEV': self.fund_data_dict['BOOK_LEVERAGE_TTM'],
#             'DTOA': self.fund_data_dict['DEBT_TO_ASSET_RATIO_TTM']
#         } 
        
#         # quality 
#         # TODO: add quality 

#         # values 
#         values = 1 / self.fund_data_dict['PB_RATIO_TTM']
#         self.style_class_dict['Values'] = {
#             'BP': values 
#         }

#         # growth 
#         self.style_class_dict['Growth'] = {
#             'EGRO': self.fund_data_dict['BASIC_EARNINGS_PER_SHARE'],  
#             # 'SGRO' 
#         }

#         # Earning Yield  
#         self.style_class_dict['EarningYield'] = {
#             'ETOP': self.fund_data_dict['EP_RATIO_TTM'], 
#             # 'EM': self.fund_data_dict['EBIT_TTM'] + 
#             'CETOP': self.fund_data_dict['CASH_EQUIVALENT_PER_SHARE_TTM']
#         }


    def calc_nl_size_fac(self):
        """
        compute nonlinear size
        """
        # (nonelinear size 是正向因子, 单因子绩效很优秀)
        # 预计耗时5分钟
        size = np.log(self.eod_data_dict["TotalMarketValue"])
        size3 = np.log(self.eod_data_dict["TotalMarketValue"]) ** 3
        size = size.values
        size3 = size3.values
        non_nan_idx = (~ np.isnan(size)) & (~ np.isnan(size3))
        nl_size = []
        for i in range(len(self.ret_df.columns)):
            non_nan_idx_i = non_nan_idx[:, i]
            if non_nan_idx_i.sum() == 0:
                nl_size.append([np.nan] * size.shape[0])
            else:
                y = size3[non_nan_idx_i, i]
                x = size[non_nan_idx_i, i]
                beta = np.cov(y, x)[0][1] / np.var(x)
                nl_size.append(size3[:, i] - beta * size[:, i])
        nl_size = pd.DataFrame(np.array(nl_size).T, index=self.ret_df.index, columns=self.ret_df.columns)
        self.style_class_dict["NLSize"] = {'nl_size': nl_size}


    @staticmethod
    def __calc_single_capm(stock_ret, index_ret, window=252):
        betas = []
        err_std = []
        alpha = []
        stock_ret_length = len(stock_ret)
        if stock_ret_length < window:
            return [([np.nan] * stock_ret_length), ([np.nan] * stock_ret_length), ([np.nan] * stock_ret_length)]
        for i in range(window, stock_ret_length):
            stk = stock_ret[i - window:i]
            idx = index_ret[i - window:i]
            non_nan_idx = ~np.isnan(stk)
            if non_nan_idx.sum() < 50: 
                beta = np.nan
            else:  
                stk_non_nan = stk[non_nan_idx]
                idx_non_nan = idx[non_nan_idx]
                beta = (np.cov(stk_non_nan, idx_non_nan) / np.var(idx_non_nan))[0, 1]
            betas.append(beta)
            err_std.append(np.nanstd(idx - stk * beta))
            alpha.append(np.nanmean(idx - stk * beta))
        betas = ([np.nan] * window) + betas
        err_std = ([np.nan] * window) + err_std
        alpha = ([np.nan] * window) + alpha
        return [alpha, betas, err_std]


    def calc_beta(self):
        # alpha 为正向因子, beta和err_std均表现为反向因子
        # 大约需要8分钟
        oo_ret_df = self.ret_df_dict['o2next_o'].shift(1, axis=1)
        masked_tmv = (oo_ret_df - oo_ret_df) + self.eod_data_dict['TotalMarketValue'].shift(1, axis=1)  # 套一层mask，保证权重相加等于1
        market_ret_df = (masked_tmv * oo_ret_df).sum() / masked_tmv.sum()
        market_ret = market_ret_df.values
        ret_nd = oo_ret_df.values
        alphas = []
        betas = []
        err_stds = []
        for i in range(len(oo_ret_df.index)):
            stock_ret = ret_nd[i, :]
            result = self.__calc_single_capm(stock_ret, market_ret)
            alphas.append(result[0])
            betas.append(result[1])
            err_stds.append(result[2])
        alphas = pd.DataFrame(alphas, index=oo_ret_df.index, columns=oo_ret_df.columns)
        betas = pd.DataFrame(betas, index=oo_ret_df.index, columns=oo_ret_df.columns)
        err_stds = pd.DataFrame(err_stds, index=oo_ret_df.index, columns=oo_ret_df.columns)
        self.style_class_dict["CAPM"] = {
            'alpha': alphas,
            'betas': betas,
            'err_std': err_stds
        }

    def get_non_extrem_and_norm(self):
        """
        remove extreme values and standardize
        """
        for class_name in self.style_class_dict.keys():
            for fac_name in self.style_class_dict[class_name].keys():
                fac = self.style_class_dict[class_name][fac_name]
                fac = fac.shift(1, axis=1)  # !!! for complex reasons this shift has to be placed here !!!
                fac[fac < fac.quantile(0.01)] = np.nan
                fac[fac > fac.quantile(0.99)] = np.nan
                if fac_name == 'nl_size' or fac_name == 'size':
                    self.style_class_dict[class_name][fac_name] = (fac - fac.mean()) / fac.std()
                else:
                    masked_tmv = self.eod_data_dict['TotalMarketValue'].shift(1, axis=1) + (fac - fac)  # add mask，理由同上
                    self.style_class_dict[class_name][fac_name] = (fac - (masked_tmv * fac).sum() / masked_tmv.sum()) / fac.std()

    def merge_class_factor(self):
        """
        aggregate style factors in each subclass with equal weights
        """
        for class_name in self.style_class_dict.keys():
            if class_name != "CAPM":
                # process size first 
                frame = self.style_class_dict['Size']['size'].fillna(1)
                frame = frame - frame
                count_frame = frame.copy()
                # for others
                for fac_name in self.style_class_dict[class_name].keys():
                    fac = self.style_class_dict[class_name][fac_name]
                    count_frame = count_frame + 1 * (~fac.isna())  # count the number of times an entry has nan
                    frame = frame + fac.fillna(0)

                fac = (frame / count_frame)
                masked_tmv = self.eod_data_dict['TotalMarketValue'].shift(
                    1, axis=1) + (fac - fac)
                self.class_factor_dict[class_name] = (
                    fac - (masked_tmv * fac).sum() / masked_tmv.sum()) / fac.std()
            # process CAPM factors
            else:
                for fac_name in self.style_class_dict[class_name].keys():
                    fac = self.style_class_dict[class_name][fac_name]
                    masked_tmv = self.eod_data_dict['TotalMarketValue'].shift(1, axis=1) + (fac - fac)
                    self.class_factor_dict[fac_name] = (
                        fac - (fac * masked_tmv).sum() / masked_tmv.sum()
                    ) / fac.std()

    # def merge_class_factor_new(self):
    #     # 将每类风格因子等权合成为大类因子
    #     for class_name in self.style_class_dict.keys():
    #         for fac_name in self.style_class_dict[class_name].keys():
    #             fac = self.style_class_dict[class_name][fac_name]
    #             self.class_factor_dict[fac_name] = (fac - (self.eod_data_dict['TotalMarketValue'] * fac).sum() # TODO: fix mask tmv
    #                                                 / self.eod_data_dict['TotalMarketValue'].sum()) / fac.std()


    # ===========================
    # ------ i/o ----------------
    # ===========================

    def saving_df(self):
        """
        save risk factors (subclasses)
        """
        # data = {}
        # for class_name in self.style_class_dict.keys():
        #     for key in self.style_class_dict[class_name].keys():
        #         data["eod_{}_{}".format(class_name, key)] = self.style_class_dict[class_name][key]
        # self.myconnector.save_eod_feature(data=data, feature_type="barra", encrypt=False,
        #                                   where=self.saving_path)
        for class_name in self.style_class_dict.keys():
            for key in self.style_class_dict[class_name].keys():
                subclass_risk_factor_df = self.style_class_dict[class_name][key]
                self.myconnector.save_eod_feature(
                    f'{class_name}_{key}',
                    subclass_risk_factor_df,
                    des='risk_factor'
                )

    def saving_merged_df(self):
        """
        save risk factors (aggregated)
        """
        # data = {}
        # for key in self.class_factor_dict.keys():
        #     data["eod_{}".format(key)] = self.class_factor_dict[key]
        # self.myconnector.save_eod_feature(data=data, feature_type="barra", encrypt=False,
        #                                   where=self.saving_path + '/class_factors')
        
        # define path 
        class_factor_save_path = self.saving_path + '/class_factors'
        if not os.path.exists(class_factor_save_path):
            os.mkdir(class_factor_save_path)
        
        # save
        for key in self.class_factor_dict.keys():
            class_factor_df = self.class_factor_dict[key]
            self.myconnector.save_eod_feature(
                key,
                class_factor_df, 
                des='risk_factor/class_factors'
            )
        
    # TODO: what is this for? Is this even necessary? 
    def save_idx_weight(self):
        """
        save index member stock weight into csv 
        """
        # compute
        for index in self.idx_list:
            index_weight_df = self.get_one_index_weight(index)
            self.idx_weight_dict[index] = index_weight_df

        # save
        path = self.saving_path + '/idx_weight_dict'
        if not os.path.exists(path):
            os.mkdir(path)
        for index in self.idx_list:
            self.idx_weight_dict[index].to_csv(path + '/' + index + '.csv')

    # =============================
    # ---------- main -------------
    # =============================

    def start_loading_data_process(self):
        """
        main process for computing the risk factor
        """
        # retrieve index member stock data 
        print("Reading Index Member Stock ...")
        t0 = time.time()
        self.save_idx_weight()
        print("Reading Index Member Stock takes ", time.time() - t0)

        # get industry and plain data 
        self.get_ind_date()
        print("Reading Plain and Industry ...")
        t0 = time.time()
        self.get_plain_data()
        print("Reading Plain and Industry takes ", time.time() - t0)

        # compute style factors
        print("Computing Style Factors ...")
        t0 = time.time()
        self.calc_style_fac()
        # self.calc_fund_style_fac() 
        self.calc_nl_size_fac()
        self.calc_beta()
        self.get_non_extrem_and_norm()
        print("Computing Style Factors takes ", time.time() - t0)

        # compute style factor risk returns 
        self.merge_class_factor()
        print("Saving Style Factors ...")
        t0 = time.time()
        self.saving_merged_df()
        print("Saving Style Factors take ", time.time() - t0)
        self.adj_class_fac()  # ??? 先adj再存?
        print("Computing Major Style Factor Returns ...")
        t0 = time.time()
        # for ret_type in self.return_type_list:
        #     self.calc_idx_ind_rtn(ret_type)
        for ret_type in self.return_type_list:
            self.calc_fac_ret(ret_type)

        print("Computing Major Style Factor Returns take", time.time() - t0)

        # saving returns
        print("Saving Returns ...")
        t0 = time.time()
        self.save_fac_ret()
        self.saving_df()
        print("Saving Returns takes", time.time() - t0)


    # ==================================
    # --------- factor returns ---------
    # ==================================

    def adj_class_fac(self):
        """
        adjust class factor 
        """
        for fac_name in self.class_factor_dict.keys():
            fac = self.class_factor_dict[fac_name]
            upper = fac.quantile(0.95)
            lower = fac.quantile(0.05)
            fac[fac > upper] = np.nan
            fac[fac < lower] = np.nan
            fac = (fac - fac.mean()) / fac.std()
            self.class_factor_dict_adj[fac_name] = fac.fillna(0)

    def calc_fac_ret(self, return_type):
        """
        compute aggregated factor returns
        """
        ret_df = self.ret_df_dict[return_type]
        shift_step = 0
        fac = ret_df
        f = []
        for t in range(len(fac.columns) - shift_step):

            Y = ret_df.iloc[:, t + shift_step]
            X = pd.DataFrame()
            for class_name in self.class_factor_dict_adj.keys():
                X[class_name] = self.class_factor_dict_adj[class_name].iloc[:, t]
            data = pd.concat([Y, X], axis=1).dropna(axis=0).values.T
            if len(data[0]) > 0:
                model = sm.OLS(data[0], data[1:].T)
                res = model.fit()
                f.append(res.params)
            else:
                f.append([np.nan for _ in range(len(self.class_factor_dict_adj.keys()))])
        factor_return_df = pd.DataFrame(f, columns=self.class_factor_dict_adj.keys())
        factor_return_df.index = list(self.class_factor_dict_adj.values())[0].columns[shift_step:]
        self.factor_return_df_dict[return_type] = factor_return_df

        # industry returns
        # TODO: 改简单点~ 说话的方式简单点~
        # TODO: 动态行业
        f = []
        for t in range(len(fac.columns) - shift_step):
            Y = ret_df.iloc[:, t + shift_step]
            X = self.ind_df.copy()
            data = pd.concat([Y, X], axis=1).dropna(axis=0).values.T
            if len(data[0]) > 0:
                model = sm.OLS(data[0], data[1:].T)
                res = model.fit()
                f.append(res.params)
            else:
                f.append([np.nan for _ in range(len(self.ind_df.columns))])
        ind_ret_df = pd.DataFrame(f, index=fac.columns[shift_step:], columns=self.ind_df.columns)
        self.ind_ret_df_dict[return_type] = ind_ret_df

        # 求板块label收益率（仅需计算kcb，cyb，和others，其余的指数可以直接读取收益率序列
        # kcb
        kcb_fac = fac.loc[fac.index.str[:3] == '688']
        kcb_ret = np.nanmean(kcb_fac, axis=0)  # 保留全nan时输出nan，同时有数的时候输出数
        # cyb
        cyb_fac = fac.loc[fac.index.str[:1] == '3']
        cyb_ret = np.nanmean(cyb_fac, axis=0)
        # others
        others_fac = np.where(self.plain_df == 6, fac, np.nan)
        others_ret = np.nanmean(others_fac, axis=0)
        plain_ret_df = pd.DataFrame(
            np.column_stack([kcb_ret, cyb_ret, others_ret]), 
            index=fac.columns[shift_step:], 
            columns=['kcb', 'cyb', 'others']
        )
        self.plain_ret_df_dict[return_type] = plain_ret_df

        # 计算全部变量对收益的解释能力
        ret_df = self.ret_df_dict[return_type]
        # tk_idx_ret_df = self.tk_idx_ret_dict[return_type]
        # alpha_ret_df = ret_df - tk_idx_ret_df

        shift_step = 0
        all_ticker = ret_df.index
        sample_data = pd.DataFrame([np.nan for _ in range(len(all_ticker))], columns=['sample'], index=all_ticker)
        f_1 = [sample_data]
        f_2 = [sample_data]
        dates = fac.columns.tolist()
        for t in range(len(fac.columns) - shift_step):
            Y = ret_df.iloc[:, t + shift_step]
            dt = dates[t+shift_step]
            # X = pd.concat([self.ind_df.copy(), real_plain_df], axis=1)
            X = self.ind_df.copy()
            for class_name in self.class_factor_dict_adj.keys():
                X[class_name] = self.class_factor_dict_adj[class_name].iloc[:, t]
            dynamic_plain_df = pd.get_dummies(self.plain_df[dt])  # 计算当日的板块one-hot
            all_df = pd.concat([Y, X, dynamic_plain_df], axis=1).dropna(axis=0)
            ticker_lst = all_df.index
            data = all_df.values.T

            if len(data[0]) > 0:
                # ind risk 回归 
                model = sm.OLS(data[0], data[1:-7].T)
                res = model.fit()
                res_df = pd.DataFrame(data[1:-7].T.dot(res.params), columns=[dt], index=ticker_lst)
                if t >= len(fac.columns)-1:
                    print(return_type, res.rsquared)
                f_1.append(res_df)
                # plain 回归
                residual = data[0] - data[1:-7].T.dot(res.params)
                model_red = sm.OLS(residual, data[-7:].T)
                residual_res = model_red.fit()
                if t >= len(fac.columns)-1:
                    print(return_type, residual_res.rsquared)
                residual_res_df = pd.DataFrame(data[-7:].T.dot(residual_res.params), columns=[dt], index=ticker_lst)
                f_2.append(residual_res_df)
            else:
                # f_1.append([np.nan for _ in range(len(X.columns))])
                # f_2.append([np.nan for _ in range(len(X.columns))])
                f_1.append(pd.DataFrame([np.nan for _ in range(len(ticker_lst))], columns=[dt], index=ticker_lst))
                f_2.append(pd.DataFrame([np.nan for _ in range(len(ticker_lst))], columns=[dt], index=ticker_lst))

        ind_risk_ra_df = pd.concat(f_1, axis=1, join='outer')
        ind_risk_ra_df = ind_risk_ra_df.drop(['sample'], axis=1)
        # ind_risk_ra_df.index = list(self.class_factor_dict_adj.values())[0].columns[shift_step:]
        plain_ra_df = pd.concat(f_2, axis=1, join='outer')
        plain_ra_df = plain_ra_df.drop(['sample'], axis=1)
        # plain_ra_df.index = list(self.class_factor_dict_adj.values())[0].columns[shift_step:]
        self.all_ind_risk_ra_df_dict[return_type] = ind_risk_ra_df
        self.all_plain_ra_df_dict[return_type] = plain_ra_df

    # def calc_idx_ind_rtn(self, return_type):
    #     self.idx_ret_df_dict[return_type].loc['kcb', :] = np.nansum(self.plain_df.loc[:, 'kcb'].values.reshape((self.ret_df_dict[return_type].values.shape[0],1)) * self.ret_df_dict[return_type].values, axis=0)/self.plain_df.loc[:, 'kcb'].sum()
    #     self.idx_ret_df_dict[return_type].loc['cyb', :] = np.nansum(self.plain_df.loc[:, 'cyb'].values.reshape((self.ret_df_dict[return_type].values.shape[0],1)) * self.ret_df_dict[return_type].values, axis=0)/self.plain_df.loc[:, 'cyb'].sum()
    #     self.idx_ret_df_dict[return_type].loc['others', :] = np.nansum(self.plain_df.loc[:, 'others'].values.reshape((self.ret_df_dict[return_type].values.shape[0],1)) * self.ret_df_dict[return_type].values, axis=0)/self.plain_df.loc[:, 'others'].sum()
    #     idx_ret_df = self.idx_ret_df_dict[return_type]

    #     exclude_plain_df = self.plain_df.copy()
    #     exclude_plain_df['000905'] = exclude_plain_df['000905'] - exclude_plain_df['kcb'] - exclude_plain_df['cyb']
    #     exclude_plain_df['000905'].replace(-1, 0, inplace=True) 
    #     exclude_plain_df['000852'] = exclude_plain_df['000852'] - exclude_plain_df['kcb'] - exclude_plain_df['cyb']
    #     exclude_plain_df['000852'].replace(-1, 0, inplace=True)
    #     exclude_plain_df = exclude_plain_df.loc[:, idx_ret_df.index]
    #     idx_index = exclude_plain_df.index
    #     idx_col = idx_ret_df.columns
    #     self.tk_idx_ret_dict[return_type] = pd.DataFrame(exclude_plain_df.values.dot(idx_ret_df.values), columns=idx_col, index=idx_index)

    def save_fac_ret(self):
        """
        save all factor returns
        """
        path_root = self.saving_path + "/return_dfs/"
        path_fac = self.saving_path + "/return_dfs/factor_return_df_dict/"
        path_idx = self.saving_path + "/return_dfs/ind_ret_df_dict/"
        path_plain = self.saving_path + "/return_dfs/plain_ret_df_dict/"
        path_ind_risk_ra = self.saving_path + "/return_dfs/path_ind_risk_ra/"
        path_plain_ra = self.saving_path + "/return_dfs/path_plain_ra/"
        for path in [path_root, path_fac, path_idx, path_plain, path_ind_risk_ra, path_plain_ra]:
            if not os.path.exists(path):
                os.mkdir(path)

        for key in self.factor_return_df_dict.keys():
            self.factor_return_df_dict[key].to_csv(path_fac + '/' + key + '.csv')
        for key in self.ind_ret_df_dict.keys():
            self.ind_ret_df_dict[key].to_csv(path_idx + '/' + key + '.csv')
        for key in self.plain_ret_df_dict.keys():
            self.plain_ret_df_dict[key].to_csv(path_plain + '/' + key + '.csv')
        for key in self.all_ind_risk_ra_df_dict.keys():
            self.all_ind_risk_ra_df_dict[key].to_csv(path_ind_risk_ra + '/' + key + '.csv')
        for key in self.all_plain_ra_df_dict.keys():
            self.all_plain_ra_df_dict[key].to_csv(path_plain_ra + '/' + key + '.csv')
        # for key in self.tk_idx_ret_dict.keys():
        #     self.tk_idx_ret_dict[key].to_csv(path_idx_rtn + '/' + key + '.csv')
