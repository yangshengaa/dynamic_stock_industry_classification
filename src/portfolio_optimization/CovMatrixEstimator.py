"""
Covariance Estimation 
"""

# load packages 
import os
import time
import datetime
import pickle
# import scipy
from numpy.linalg import eigh
# import logging
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp 
import statsmodels.api as sm
# 控制进程数
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings('ignore')

# initialize dataserver 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

# load files 
import src.portfolio_optimization.config as cfg


class CovMatrixEstimator:
    def __init__(self):
        # 数据存放
        self.factor_return_df_dict = {}  # 存放大类风格因子收益率
        self.factor_cov_dict = {}        # 存放风格因子协方差矩阵
        self.idio_return_df_dict = {}    # 存放特质收益率
        self.idio_var_dict = {}          # 存放特质收益方差
        self.class_factor_dict_adj = {}  # 存放大类风格因子值

        self.factor_cov_raw_dict = {}
        self.idio_var_raw_dict = {}

        # 其他常量
        self.class_name = cfg.class_name
        self.return_type_list = cfg.return_type_list    
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date

        # dynamic ind 
        self.use_dynamic_ind = cfg.use_dynamic_ind
        self.dynamic_ind_name = cfg.dynamic_ind_name
        
        # path 
        self.cov_save_path = cfg.cov_save_path   # 协方差矩阵写入路径     
        self.ret_read_path = cfg.ret_save_path   # 因子收益/特质收益读取路径
        self.class_factor_read_path = cfg.class_factor_path
        
        # read parameters
        # Raw
        self.lam = cfg.lam  # EWMA权重
        self.h = cfg.h  # 样本时间周期
        # Newey West Related
        self.N_W = cfg.N_W
        self.D = cfg.D
        self.pred_period = cfg.pred_period
        # Monte-Carlo Simulation 
        self.alpha = cfg.alpha # 调整系数，取值（1，2）之间
        # Bayesian Shrinkage 
        self.q = cfg.q
        self.bs_group_num = cfg.bs_group_num
        # Volatility Regime Adjustment 
        self.h_vol = cfg.h_vol
        self.lam_vol = cfg.lam_vol
        # structural adjustment (结构化调整)
        self.h_struc = cfg.h_struc
        self.min_o = cfg.min_o
        self.E0 = cfg.E0


    def load_history(self):
        # data 
        self.myconnector = PqiDataSdkOffline()
        self.all_stocks = self.myconnector.get_ticker_list()
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers=self.all_stocks, 
            start_date=self.start_date,
            end_date=self.end_date,
            fields=['FloatMarketValue']
        )
        self.date_list = list(self.eod_data_dict['FloatMarketValue'].columns)
        self.tickers = list(self.eod_data_dict["FloatMarketValue"].index) 
        self.index_code_to_name = cfg.index_code_to_name


    def read_ret_data(self, return_type):
        """ read a factor return and idio returns for a specific return type  """
        # TODO: change to other formats 
        ret_save_path = os.path.join(self.ret_read_path, self.dynamic_ind_name)
        factor_return = pd.read_feather('{}/Factor_return_{}_'.format(ret_save_path, return_type)).set_index('index')
        idio_return = pd.read_feather('{}/Idio_return_{}_'.format(ret_save_path, return_type)).set_index('index')
        factor_return.index = factor_return.index.astype(int)

        # algin tickers
        idio_return.index = [str(x) for x in idio_return.index]
        idio_return = (idio_return.T + (self.eod_data_dict['FloatMarketValue'] - self.eod_data_dict['FloatMarketValue'])).T  
        idio_return.index = [int(x) for x in idio_return.index]
        return factor_return, idio_return


    def load_ret_data(self):
        """ read factor/idio returns for all types """
        for name in self.return_type_list:
            self.factor_return_df_dict[name], self.idio_return_df_dict[name] = self.read_ret_data(name)


    def get_cov_dates(self):
        """ get dates (take intersection) """ 
        factor_ret = self.factor_return_df_dict[self.return_type_list[0]]
        date_list_ret = [str(x) for x in list(factor_ret.index)]
        self.date_list_cov = list(set(date_list_ret) & set(self.date_list))
        self.date_list_cov.sort()


    def read_factor_data(self, feature_name, tickers, date_list):
        """ 
        read a single aggregated style factor 
        :param feature_name: style factor name
        """
        factor_df = self.myconnector.read_eod_feature(
            feature_name, des='risk_factor/class_factors', dates=date_list
        )
        return factor_df 


    def load_factor_data(self):
        """
        read all style factors
        """
        for name in self.class_name:
            self.class_factor_dict_adj[name] = self.read_factor_data(name, self.tickers, self.date_list)


    def get_ind_data(self):
        """ get static industry data """
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
    
    def get_dynamic_ind_data(self):
        """ get dynamic industry data """
        ind_df = self.myconnector.read_ind_feature(self.dynamic_ind_name)
        ind_df_selected = ind_df[self.date_list]
        self.ind_df = ind_df_selected

    # ===================================================================
    # ================= 原始协方差估计 ===================================
    # ===================================================================

    def raw_cov_est(self, return_type):
        """ 因子收益率和特质收益率raw协方差矩阵计算(采用EWMA加权) """
        # 统一储存格式 str不是int
        # factor_return_df = self.factor_return_df_dict[return_type]
        # idio_return_df = self.idio_return_df_dict[return_type]

        # 数据准备
        date_list_cov_int = [int(x) for x in self.date_list_cov]

        factor_return_df = self.factor_return_df_dict[return_type].loc[date_list_cov_int,:] 
        idio_return_df = self.idio_return_df_dict[return_type].loc[date_list_cov_int,:]
        temp_factor_cov = {}
        temp_idio_var = {}
        # date_list = factor_return_df.index
 
        self.factor_num = factor_return_df.shape[1]
        self.ticker_num = idio_return_df.shape[1]
        total_date = len(self.date_list_cov)


        weight = self.lam ** np.arange(self.h)[::-1] 
        weight = weight / weight.sum()   # 预先归一化
        weight = np.array([weight]).T    # 转换成column vector        
        
        # pool = mp.Pool(processes=200)
        # process_list = []
        # rolling 计算每日cov
        for current_t in range(self.h, (total_date+1)):
            # 提取当前rolling期内的相关数据
            f_current = factor_return_df[(current_t - self.h):current_t].values
            idio_current = idio_return_df[(current_t - self.h):current_t].values

            # if all nan, skip to speed up 
            # if np.isnan(f_current).all() and np.isnan(idio_current).all():
            #     temp_factor_cov[self.date_list_cov[current_t - 1]] = f_current   
            #     temp_idio_var[self.date_list_cov[current_t - 1]] = pd.DataFrame([idio_current], columns=idio_return_df.columns)
            #     continue

            # 因子收益率
            f_current_weighted = f_current * weight 
            f_current_demeaned = f_current - f_current_weighted.mean()
            factor_cov = (f_current_demeaned * weight).T @ f_current_demeaned

            if self.N_W:
                for delta in range(1, self.D + 1):
                    adjusted_factor_cov_D = (f_current_demeaned[delta:] * weight[delta:]).T @ f_current_demeaned[:-delta]
                    factor_cov = factor_cov + (1 - delta / (self.D + 1)) * (adjusted_factor_cov_D + adjusted_factor_cov_D.T)
                    # 此处的调整的权重为Bartlett Taper, 可以调整为其他 (any weight function with a nonnegative spectral density estimate)
                factor_cov = factor_cov * self.pred_period 

            # Check for psd  
            # print(self.date_list[current_t - 1])
            # print(np.min(scipy.linalg.eigh(factor_cov)[0]))

            # 特质收益率
            idio_current_weighted = idio_current * weight 
            idio_nan_mask = np.isnan(idio_current_weighted).sum(axis=0) > (self.h - 50)  # 仅当至少有50个非nan的票的时候才有数
            idio_current_demeaned = idio_current - np.nanmean(idio_current_weighted, axis=0)
            idio_var = np.nansum(weight * idio_current_demeaned * idio_current_demeaned, axis=0)
            idio_var[idio_nan_mask] = np.nan

            if self.N_W:
                for delta in range(1, self.D + 1):
                    idio_var_adjust = np.nansum(weight[delta:] * idio_current_demeaned[delta:] * idio_current_demeaned[:-delta], axis=0)
                    idio_var = idio_var + (2 * idio_var_adjust) * (1 - delta / (self.D + 1)) 
                idio_var[idio_var < 0] = 0  # 相当于投影到最近的psd对角矩阵  # TODO: 不能确保>=0（有无更好的处理方式？）
                idio_var = idio_var * self.pred_period 
            temp_factor_cov[self.date_list_cov[current_t - 1]] = factor_cov   
            temp_idio_var[self.date_list_cov[current_t - 1]] = pd.DataFrame([idio_var], columns=idio_return_df.columns)
            
        self.factor_cov_raw_dict[return_type] = temp_factor_cov
        self.idio_var_raw_dict[return_type] = temp_idio_var

    # ===================================================================
    # ================= 特征值调整 =======================================
    # ===================================================================

    @staticmethod
    def eigen_adj_est_each_period(curr_factor_cov, factor_num, h, alpha):
        """
        特征值调整子进程: 蒙特卡洛模拟
        :param curr_factor_cov: 当前期内的蒙特卡罗模拟拟调整的矩阵
        :return 调整完的矩阵
        """
        # speed hack: if all nan, return itself, as if not adjusted
        if np.isnan(curr_factor_cov).all():
            return curr_factor_cov
        
        e_true_values, e_true_vectors = eigh(curr_factor_cov)

        # bootstrap
        bootstrap_num = 500  # boostrap数量
        # dev = np.zeros((bootstrap_num, factor_num))
        dev_list = []
        
        for m in range(bootstrap_num):
            # 生成模拟特征因子收益矩阵
            b_m = np.random.normal(
                loc=0, 
                scale=np.sqrt(np.array([e_true_values]).T), 
                size=(factor_num, h)
            )  # 每一行都是同一个标准差

            r_m = e_true_vectors @ b_m  # 模拟因子收益矩阵
            r_m_demeaned = r_m - r_m.mean(axis=1, keepdims=True)  
            cov_b = r_m_demeaned @ r_m_demeaned.T / h

            e_pseudo_value, e_pseudo_vector = eigh(cov_b)
            e_diag_true = np.diag(e_pseudo_vector.T @ curr_factor_cov @ e_pseudo_vector)
            quotient = e_diag_true / e_pseudo_value  # 模拟风险偏差
            # dev[m,:] = quotient
            dev_list.append(quotient)
        
        dev = np.vstack(dev_list)  # concat 
        mean_dev = np.sqrt(dev.mean(axis = 0))  
        final_dev = alpha * (mean_dev - 1) + 1
        cov_eigen_adj = e_true_vectors @ np.diag((final_dev * final_dev) * e_true_values) @ e_true_vectors.T
        return cov_eigen_adj


    def eigen_adj_est(self, return_type):
        """ 因子收益率协方差矩阵特征值调整"""
        temp_factor_cov = self.factor_cov_raw_dict[return_type]
        dates = list(temp_factor_cov.keys())
        adj_factor_cov = {}

        # 将每一期raw估计的协方差矩阵放入子进程蒙特卡洛
        pool = mp.Pool(processes=8)
        process_list = []
        for current_t in dates:
            process_list.append(pool.apply_async(
                self.eigen_adj_est_each_period, 
                args=(temp_factor_cov[current_t], self.factor_num, self.h, self.alpha)
                )
            )
        time.sleep(1)
        for i in range(len(process_list)):
            adj_factor_cov[dates[i]] = process_list[i].get()
        pool.close()
        pool.join()
        self.factor_cov_raw_dict[return_type] = adj_factor_cov

    # ===================================================================
    # ================= 贝叶斯调整 =======================================
    # =================================================================== 
    
    @staticmethod
    def bs_adj_est_each_period(today_size, today_idio_var, bs_group_num, q):
        """
        贝叶斯调整子进程
        :param today_idio_var: 今日特质收益协方差
        :param today_size: 今日截面个股市值
        :return 贝叶斯调整后的特质协方差
        """
        # print(today_size, today_idio_var)
        today_size_group = pd.qcut(today_size, q=bs_group_num, labels=range(bs_group_num))
        idio_bs_list = []
        # 对每一组组内做贝叶斯压缩
        for g in range(bs_group_num):
            weight_g = today_size[today_size_group == g]
            idio_sig_g = np.sqrt(today_idio_var.loc[:, today_size_group == g])  
            wid = weight_g * idio_sig_g
            idio_mean_g = (wid.sum(axis = 1) / (~wid.isna() * weight_g).sum(axis = 1)).item() # 组内加权平均
            delta_sig_g = np.sqrt(((idio_sig_g - idio_mean_g) ** 2).sum(axis = 1) / (~wid.isna()).sum(axis = 1)).item() # 组内偏离程度

            qabs = abs(idio_sig_g - idio_mean_g) * q
            eta = qabs / (delta_sig_g + qabs)  # 贝叶斯收缩权重

            idio_sig_bs_g = eta * idio_mean_g + (1 - eta) * idio_sig_g
            idio_bs_list.append(idio_sig_bs_g)
        
        idio_bs = pd.concat(idio_bs_list, axis=1)                
        idio_bs[list(set(today_size.index) - set(idio_bs.columns))] = np.nan # 其他未分组的地方放nan 
        idio_bs = idio_bs.sort_index(axis = 1)        # 对columns日期进行排序
        idio_var_new = idio_bs * idio_bs
        return idio_var_new
    
    def bs_adj_est(self, return_type):
        """ 特质收益率协方差矩阵贝叶斯调整 """
        # 读取协方差
        temp_idio_var = self.idio_var_raw_dict[return_type]

        idio_var_bs = {}
        size = self.eod_data_dict['FloatMarketValue']
        date_list_bs = list(temp_idio_var.keys())

        # 读取收益率
        pool = mp.Pool(processes=8)
        process_list = []
        for current_t in date_list_bs:
            today_size, today_idio_var = size[current_t], temp_idio_var[current_t]
            process_list.append(pool.apply_async(
                self.bs_adj_est_each_period, 
                args=(today_size, today_idio_var, self.bs_group_num, self.q)
            ))
        time.sleep(1)
        for i in range(len(process_list)):
            idio_var_bs[date_list_bs[i]] = process_list[i].get()
        pool.close()
        pool.join()
        self.idio_var_raw_dict[return_type] = idio_var_bs


    # ===================================================================
    # ================= 波动率调整 =======================================
    # =================================================================== 


    def vol_adj_est(self, return_type):
        """ 因子和特质收益率协方差矩阵波动率调整"""
       # TODO: 测试更短时间周期和半衰期的效果
        temp_factor_cov = self.factor_cov_raw_dict[return_type]
        temp_idio_var = self.idio_var_raw_dict[return_type]

        vol_adj_factor_cov = {}

        weight_vol = self.lam_vol ** np.arange(self.h_vol)[::-1]
        weight_vol = weight_vol / weight_vol.sum()  # 预先归一化
        
        date_list_vol_cal = list(temp_idio_var.keys())
        date_list_vol_cal_int = [int(x) for x in date_list_vol_cal]
        factor_return_df = self.factor_return_df_dict[return_type].loc[date_list_vol_cal_int,:] 
        factor_return_nextday = factor_return_df.shift(-1)
        factor_return_nextday_np = factor_return_nextday.to_numpy()

        idio_return_df = self.idio_return_df_dict[return_type].loc[date_list_vol_cal_int,:] 
        idio_return_nextday = idio_return_df.shift(-1)
        idio_return_nextday_np = idio_return_nextday.to_numpy()

        new_idio_var = {}
        size = self.eod_data_dict['FloatMarketValue']
        size_np =  size.to_numpy().T

        total_date = len(date_list_vol_cal)

        # 预先计算过往pred_period的风险偏误统计量   
        # TODO: 多进程计算
        bias_f_agg, bias_idio_agg = [], [] 
        for current_t in range(1 + self.pred_period, total_date):
            # 因子收益波动
            # bias_f_agg.append(
            #     np.sqrt(
            #     ((
            #         ((factor_return_nextday.iloc[(current_t - 1 - self.pred_period):current_t - 1,:] + 1).prod() - 1) 
            #       / np.sqrt(np.diag(temp_factor_cov[date_list_vol_cal[(current_t - 1 - self.pred_period)]]))) ** 2).mean())
            # )

            # # 特质收益波动
            # bias_idio_each = (((idio_return_nextday.iloc[(current_t - 1 - self.pred_period):current_t - 1,:] + 1).prod() - 1) 
            #            / np.sqrt(temp_idio_var[date_list_vol_cal[(current_t - 1 - self.pred_period)]])) ** 2
            cur_date = date_list_vol_cal[(current_t - 1 - self.pred_period)]
            # 因子收益波动
            bias_f_agg.append(
                np.sqrt(
                ((
                    (np.nanprod(factor_return_nextday_np[(current_t - 1 - self.pred_period):current_t - 1] + 1, axis=0) - 1) 
                  / np.sqrt(np.diag(temp_factor_cov[cur_date]))) ** 2).mean())
            )

            # # 特质收益波动
            bias_idio_each = ((np.nanprod(idio_return_nextday_np[(current_t - 1 - self.pred_period):current_t - 1] + 1, axis=0) - 1) 
                       / np.sqrt(temp_idio_var[cur_date])) ** 2

            bias_idio_each = bias_idio_each + bias_idio_each * 0 
            # bias_idio_each[bias_idio_each >= 100] = 1
            bias_idio_each = np.where(bias_idio_each >= 100, 1, bias_idio_each)
            bias_idio_agg.append(bias_idio_each)
        bias_f_agg = np.array(bias_f_agg)

        # rolling调整
        for current_t in range((self.h_vol + self.pred_period), (total_date)):

            bias_f = bias_f_agg[current_t - self.h_vol - self.pred_period : current_t - self.pred_period]
            u_s = np.vstack(bias_idio_agg[current_t - self.h_vol - self.pred_period : current_t - self.pred_period]) 
            # curr_period_size = size.loc[:, date_list_vol_cal[(current_t - self.h_vol)]: date_list_vol_cal[(current_t - 1)]].T.values
            curr_period_size = size_np[current_t - self.h_vol : current_t]
            tus = curr_period_size * u_s 
            bias_idio = np.sqrt(np.nansum(tus, axis=1) / np.nansum(~np.isnan(tus) * curr_period_size, axis=1))  # 市值加权算横截面特质风险偏误统计量

            lambda_f = ((bias_f * bias_f) * weight_vol).sum()           # 因子波动率乘数
            lambda_idio = ((bias_idio * bias_idio) * weight_vol).sum()  # 特质波动率乘数

            vol_adj_factor_cov[date_list_vol_cal[current_t - 1]] = lambda_f * temp_factor_cov[date_list_vol_cal[current_t - 1]]
            new_idio_var[date_list_vol_cal[current_t - 1]] = lambda_idio * temp_idio_var[date_list_vol_cal[current_t - 1]] 

        
        self.factor_cov_raw_dict[return_type] = vol_adj_factor_cov
        self.idio_var_raw_dict[return_type] = new_idio_var




    # ===================================================================
    # ================= 结构化调整 =======================================
    # =================================================================== 


    def struc_adj_est(self, return_type):
        idio_var_est = self.idio_var_raw_dict[return_type]
        new_idio_var = {}
        
        idio_return = self.idio_return_df_dict[return_type]
        idio_return_df = idio_return.T
        idio_return_df.columns = [str(x) for x in idio_return_df.columns]
        sig_u = (idio_return_df.rolling(self.h_struc, min_periods= self.min_o, axis = 1).quantile(0.75) - idio_return_df.rolling(self.h_struc, min_periods= self.min_o, axis = 1).quantile(0.25)) / 1.35
        z_u = abs(idio_return_df.rolling(self.h_struc, min_periods= self.min_o, axis = 1).std() / sig_u - 1)
        gamma = z_u.applymap(lambda x: min(np.exp(1 - x), 1))


        date_list_struc = list(set(list(idio_return_df.columns[self.h_struc - 1:])) & set(list(idio_var_est.keys())))
        date_list_struc.sort()

        # static industry: copy once and replace columns in the loop 
        if not self.use_dynamic_ind:
            X = self.ind_df.copy()  # 外层copy一次，循环内自己换factor_name
        else:
            sample_cross_section = self.ind_df.iloc[:, -1]
            ind_labels = sorted(list(set(sample_cross_section.tolist())))
            map_dict = dict(zip(ind_labels, np.eye(len(ind_labels), dtype=int)))


        for date in date_list_struc:
            today_original_idio_var = idio_var_est[date]
            sig_ts = np.sqrt(today_original_idio_var[gamma[(gamma == 1)[date]].index])

            # 对gamma=1的股票，回归计算拟合系数
            # dynamic industry: extract days and one hot encode it 
            if self.use_dynamic_ind:
                X = pd.DataFrame(
                    self.ind_df[date].apply(lambda x: map_dict[x]).tolist(),
                    columns=ind_labels,
                    index=self.all_stocks
                )
            # X = self.ind_df.copy()
            for class_name in self.class_factor_dict_adj.keys():
                X[class_name] = self.class_factor_dict_adj[class_name][date]

            X_ts = X.T[sig_ts.columns].T 
            today_size = self.eod_data_dict['FloatMarketValue'][date][sig_ts.columns]
            all_df = pd.concat([np.log(sig_ts).T, X_ts, today_size], axis=1).dropna(axis = 0)
            data = all_df.values.T
            # TODO: handle zero size array
            data_x, data_y, data_weight = data[1:-1].T, data[0], data[-1]
            if len(data_y) > 0:
                model = sm.WLS(data_y, data_x, weights=data_weight)
                res = model.fit()
                params = res.params
            else:
                params = np.array([np.nan] * (data.shape[0] - 2))

            # 对gamma<1的股票，做结构化调整
            adjust_stock = today_original_idio_var[gamma[(gamma < 1)[date]].index].columns
            X_adjust = X.T[adjust_stock].T
            sig_str = self.E0 * np.exp(X_adjust @ params)
            sigma_adjust = (1 - gamma[date][adjust_stock]) * sig_str + gamma[date][adjust_stock] * np.sqrt(today_original_idio_var[adjust_stock])

            today_original_idio_var[adjust_stock] = (sigma_adjust * sigma_adjust)
            new_idio_var[date] = today_original_idio_var

        self.idio_var_raw_dict[return_type] = new_idio_var


    # TODO：后续待添加：其他协方差矩阵估计方式


    def cal_cov(self):
        """ 协方差矩阵计算和调整 """
        for ret_type in self.return_type_list:
            print('正在计算样本协方差矩阵' + ret_type)
            t0 = time.time()
            self.raw_cov_est(ret_type)
            print("计算样本协方差矩阵耗时", time.time() - t0)

            if cfg.eigen_adj:
                t0 = time.time()
                print('正在进行特征值调整' + ret_type)
                self.eigen_adj_est(ret_type)
                print("特征值调整耗时", time.time() - t0)

            if cfg.struc_adj:
                t0 = time.time()
                print('正在进行结构化调整' + ret_type)
                self.struc_adj_est(ret_type)
                print("结构化调整耗时", time.time() - t0)

            if cfg.bs_adj:
                t0 = time.time()
                print('正在进行贝叶斯调整' + ret_type)
                self.bs_adj_est(ret_type)
                print("贝叶斯调整耗时", time.time() - t0)
            
            if cfg.vol_adj:
                t0 = time.time()
                print('正在进行波动率调整' + ret_type)
                self.vol_adj_est(ret_type)
                print("波动率调整耗时", time.time() - t0)
        

        self.factor_cov_dict = self.factor_cov_raw_dict
        self.idio_var_dict = self.idio_var_raw_dict


    def save_cov(self):
        '''
        把协方差矩阵储存到本地
        '''
        for return_type in self.return_type_list:
            cov_save_path = os.path.join(self.cov_save_path, self.dynamic_ind_name)
            if not os.path.isdir(cov_save_path):
                os.mkdir(cov_save_path)

            factor_cov_file_name = os.path.join(cov_save_path, 'factor_cov_est_{}'.format(return_type))
            idio_var_file_name = os.path.join(cov_save_path, 'idio_var_est_{}'.format(return_type))

            if cfg.N_W:
                factor_cov_file_name = factor_cov_file_name + '_NW{}_{}'.format(self.pred_period, cfg.D)
                idio_var_file_name = idio_var_file_name + '_NW{}_{}'.format(self.pred_period, cfg.D)
            
            if cfg.eigen_adj:
                factor_cov_file_name = factor_cov_file_name + '_Eigen'
                idio_var_file_name = idio_var_file_name + '_Eigen'

            if cfg.struc_adj:
                factor_cov_file_name = factor_cov_file_name + '_struc_{}'.format(self.h_struc)
                idio_var_file_name = idio_var_file_name + '_struc_{}'.format(self.h_struc)

            if cfg.bs_adj:
                factor_cov_file_name = factor_cov_file_name + '_bs'
                idio_var_file_name = idio_var_file_name + '_bs'

            if cfg.vol_adj:
                factor_cov_file_name = factor_cov_file_name + '_vol{}_{}'.format(cfg.h_vol, cfg.tau_vol)
                idio_var_file_name = idio_var_file_name + '_vol{}_{}'.format(cfg.h_vol, cfg.tau_vol)

            print(factor_cov_file_name)
            print(idio_var_file_name)


            with open(factor_cov_file_name + '.pkl', 'wb') as f:
                pickle.dump(self.factor_cov_dict[return_type], f)
            
            with open(idio_var_file_name + '.pkl', 'wb') as f:
                pickle.dump(self.idio_var_dict[return_type], f)


    def start_cal_cov_process(self):
        print("正在读取收益率数据")
        t0 = time.time()
        self.load_history()
        self.load_ret_data()
        self.get_cov_dates()
        self.load_factor_data()
        
        # industry 
        if self.use_dynamic_ind: # dynamic 
            self.get_dynamic_ind_data()
        else:                    # static
            self.get_ind_data()

        print("读取收益率数据耗时", time.time() - t0)

        print("正在计算协方差矩阵")
        t0 = time.time()
        self.cal_cov()
        print("计算协方差矩阵", time.time() - t0)
        
        # 数据结果储存
        print("正在储存数据")
        t0 = time.time()
        self.save_cov()
        print("储存数据耗时", time.time() - t0)
