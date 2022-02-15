"""
权重优化
"""
# load packages 
import os
import time
import pickle
import traceback
import logging
import numpy as np
import pandas as pd
from typing import Dict, List 

import cvxopt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# Control numpy threads
# os.environ["MKL_NUM_THREADS"] = "10" 
# os.environ["NUMEXPR_NUM_THREADS"] = "10" 
# os.environ["OMP_NUM_THREADS"] = "10"
# os.environ["OPENBLAS_NUM_THREADS"] = "10"

# initialize dataserver 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

# load config
import src.portfolio_optimization.config as cfg

class WeightOptimizer():
    def __init__(self):
        # basics 
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        
        # dataserver
        self.myconnector = PqiDataSdkOffline()

        # stock pools and dates
        self.all_stocks = self.myconnector.get_ticker_list()
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers=self.all_stocks, 
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.return_type_list = cfg.return_type_list
        self.trade_dates = self.myconnector.select_trade_dates(
            start_date=self.start_date, 
            end_date=self.end_date
        )

        # to be populated 
        self.factor_cov_dict = {}
        self.idio_var_dict = {}
        self.cov_save_path = cfg.cov_save_path

        self.class_factor_dict_adj = {}

        self.tickers = list(self.eod_data_dict["ClosePrice"].index)  
        self.date_list = list(self.eod_data_dict['ClosePrice'].columns)
        self.index_code_to_name = cfg.index_code_to_name

        self.sigma_holding_dict = {}
        self.holding_stock_list_dict = {}
        self.benchmark_index_list_dict = {}

        self.opt_signal_dict = {}
        
        # parameters
        self.class_name = cfg.class_name
        self.penalty_lambda = cfg.penalty_lambda
        self.penalty_theta = cfg.penalty_theta
        self.penalty_nu = cfg.penalty_nu
        self.penalty_xi = cfg.penalty_xi
        self.all_turnover_limit = cfg.turnover_limit
        self.all_style_low_limit = cfg.style_low_limit
        self.all_style_high_limit = cfg.style_high_limit
        self.all_ind_low_limit = cfg.ind_low_limit
        self.all_ind_high_limit = cfg.ind_high_limit
        self.weight_low = float(cfg.weight_low)
        self.weight_high = float(cfg.weight_high)
        self.qp_method = cfg.qp_method

        # dynamic ind 
        self.use_dynamic_ind = cfg.use_dynamic_ind
        self.dynamic_ind_name = cfg.dynamic_ind_name


    # ==============================================
    # -------------- data prep ---------------------
    # ==============================================

    def read_cov_data(self, return_type):
        """ 
        read factor cov and idio cov
        读取因子和特质收益率协方差矩阵 
        """
        if self.use_dynamic_ind:
            save_path = os.path.join(self.cov_save_path, self.dynamic_ind_name)
        else:
            save_path = self.cov_save_path
        factor_cov_file_name = os.path.join(save_path, 'factor_cov_est_{}{}'.format(return_type,cfg.adj_method))
        idio_var_file_name = os.path.join(save_path, 'idio_var_est_{}{}'.format(return_type,cfg.adj_method))

    
        f1 = open(factor_cov_file_name + '.pkl','rb')
        factor_cov = pickle.load(f1)
        f1.close()
        f2 = open(idio_var_file_name + '.pkl','rb')
        idio_var = pickle.load(f2)
        f2.close()

        return factor_cov, idio_var


    def load_cov_data(self):
        """ 
        load cov estimates of all return modes
        读取所有协方差估计 
        """
        for name in self.return_type_list:
            self.factor_cov_dict[name], self.idio_var_dict[name] = self.read_cov_data(name)


    def load_signal_data(self):
        self.input_signal = pd.read_feather(
            os.path.join(cfg.input_signal_path, cfg.input_signal_df_name)
        ).set_index('index')


    def read_factor_data(self,feature_name, tickers, date_list):
        """ 读取因子 """
        
        # path = cfg.class_factor_path
        
        # feature_name = "eod_" + feature_name
        # factor = self.myconnector.get_eod_feature(fields=feature_name, where=path, tickers=tickers, dates=date_list)
        factor_df = self.myconnector.read_eod_feature(
            feature_name, des='risk_factor/class_factors', dates=date_list
        )
        factor_df = factor_df.shift(-1, axis=1)     # * 转换回今日的风格因子值
        # return factor[feature_name].to_dataframe().shift(-1, axis = 1)  
        return factor_df


    def load_factor_data(self):
        for name in self.class_name:
            self.class_factor_dict_adj[name] = self.read_factor_data(name, self.tickers , self.date_list)

    # -------- static ind ---------- 
    def get_ind_date(self):
        """ read industry data """
        # 读取行业数据
        # # TODO: add others
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

    # --------- dynamic ind -----------
    def get_dynamic_ind_data(self):
        """ get dynamic industry data """
        ind_df = self.myconnector.read_ind_feature(self.dynamic_ind_name)
        ind_df_selected = ind_df[self.date_list]
        self.ind_df = ind_df_selected

    def prepare_dynamic_ind_data_ohe(self):
        """ determine mapping rule for the dynamic ind throughout """
        sample_cross_section = self.ind_df.iloc[:, 0]
        ind_labels = sorted(list(set(sample_cross_section.tolist())))
        map_dict = dict(zip(ind_labels, np.eye(len(ind_labels), dtype=int)))

        # append to self for later reuse
        self.ind_labels = ind_labels
        self.num_dyn_ind = len(self.ind_labels)
        self.map_dict = map_dict  

    def dynamic_ind_data_ohe(self, date: str, selected_stocks: List[str or int]):
        """ One-hot encode the dynamic industry data of one date """
        X = pd.DataFrame(
            self.ind_df[date].loc[selected_stocks].apply(lambda x: self.map_dict[x]).tolist(),
            columns=self.ind_labels,
            index=selected_stocks
        )
        return X 

    def read_ml_factor_data(self, feature_name, tickers, date_list):
        """ read ml predicted returns (factor) 读取ml预测收益率因子数据 """
        # 读取原始因子值
        factor_df = self.myconnector.read_eod_feature(
            feature_name, des='ml_factor', dates=date_list
        )
        return factor_df

    def get_opt_dates(self):
        """
        get date list (date intersection of signals, cov estimates, and basic data)
        获取需要组合权重优化的日期列表（持仓信号和协方差矩阵的日期交集）
        """  
        factor_cov_est = self.factor_cov_dict[self.return_type_list[0]]
        self.date_list_cov = [str(x) for x in list(set(factor_cov_est.keys()))]
        self.date_list_signal = list(set(self.input_signal.columns))
        self.date_list_opt = list(set(self.date_list_cov) & set(self.date_list_signal) & set(self.date_list))
        self.date_list_opt.sort()
        self.date_list_opt = self.date_list_opt[3:]  # !!!

    def prepare_data(self):
        self.load_cov_data()
        self.load_signal_data()
        self.load_factor_data()

        # dynamic ind 
        if self.use_dynamic_ind:
            self.get_dynamic_ind_data()
        # static ind 
        else:
            self.get_ind_date()
        self.get_opt_dates()

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


    def get_index_mask(self,index_list):
        """
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

    # =====================================================
    # ------------------ compute sigma --------------------
    # =====================================================

    def cal_sigma_holding(self, return_type):
        """
        compute daily holding sigma 
        计算每日持仓股票的协方差矩阵Sigma
        """
        factor_cov_est = self.factor_cov_dict[return_type]
        idio_var_est = self.idio_var_dict[return_type]

        self.benchmark_index_weight = self.get_stock_weight(cfg.benchmark_index) / 100
        self.benchmark_index_weight_filled = self.benchmark_index_weight.fillna(0)
        
        sigma_holding = {}

        if not self.use_dynamic_ind:
            X = self.ind_df.copy()
        else:
            self.prepare_dynamic_ind_data_ohe()

        for date in self.date_list_opt:
            current_date_input_signal = self.input_signal[date]
            
            # 当日持仓股票
            holding_stock_list = [x.zfill(6) for x in [str(x) for x in list(current_date_input_signal[current_date_input_signal > 1e-6].index)]]  # !!!
            self.holding_stock_list_dict[date] = holding_stock_list

            # 当日指数成分股
            self.benchmark_index_list_dict[date] = self.benchmark_index_weight[self.benchmark_index_weight[date].notna()].index
            
            # factor cov and idio cov 
            Sigma_F = factor_cov_est[date]
            diag_E = idio_var_est[date][holding_stock_list].values[0]
            diag_E[np.isnan(diag_E)] = np.nanmean(diag_E)  # TODO: 改进处理nan的方法
            # print(date)
            # print(holding_stock_list)
            # print(idio_var_est[date])
            # print(diag_E)
            # diag_E = idio_var_est[date]
            # diag_E[np.isnan(diag_E)] = np.nanmean(diag_E)  
            # diag_E = diag_E[holding_stock_list].values[0]
            Sigma_E = np.diag(diag_E)             
            
            # dynamic industry: extract days and one hot encode it 
            if self.use_dynamic_ind:  # TODO: pack into a method 
                X = pd.DataFrame(
                    self.ind_df[date].apply(lambda x: self.map_dict[x]).tolist(),
                    columns=self.ind_labels,
                    index=self.tickers
                )

            # factor loading
            for class_name in self.class_factor_dict_adj.keys():
                X[class_name] = self.class_factor_dict_adj[class_name][date]

            X_holding = X.T[holding_stock_list].fillna(0)  # TODO: 改进处理nan的方法
            Sigma_Signal = np.matmul(np.matmul(X_holding.T, Sigma_F).values, X_holding.values)

            Sigma_holding_stock = Sigma_Signal + Sigma_E

            sigma_holding[date] = Sigma_holding_stock
        
        self.sigma_holding_dict[return_type] = sigma_holding

    # =====================================================
    # ------------------ optimization ---------------------
    # =====================================================

    def qp_method_1(self, date, Sigma_holding_stock, prev_holdings):
        num_holding = Sigma_holding_stock.shape[0]
        holding_stock_list = self.holding_stock_list_dict[date]

        weight_low = self.weight_low
        weight_high = self.weight_high
        turnover_limit = self.all_turnover_limit
        style_low_limit = self.all_style_low_limit
        style_high_limit = self.all_style_high_limit   
        ind_low_limit = self.all_ind_low_limit
        ind_high_limit = self.all_ind_high_limit
        # previous_day_signal = opt_signal.shift(1, axis = 1).fillna(0)[date][holding_stock_list].values

        # specify base 
        if 'min_var' in self.obj_func:

            P_raw = 2 * matrix(Sigma_holding_stock)
            q_raw = matrix(np.zeros(num_holding))

        elif 'ret_var' in self.obj_func:  # 最大化经风险调整后收益
            P_raw = 2 * matrix(Sigma_holding_stock)

            # 预测收益率
            current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0) # TODO: 改进处理nan的方法
            q_raw = matrix(-current_holding_pred_return / self.penalty_lambda)
        else:
            raise NotImplementedError('Cannot support this objective function')

        # TODO: sparse matrix
        # 约束条件：持仓权重上下限 base quadratic functions 
        G_raw = matrix(np.vstack((np.identity(num_holding), -np.identity(num_holding))))
        h_raw = matrix(np.hstack((np.full(num_holding, weight_high), np.full(num_holding, -weight_low))))
        #h = matrix(np.hstack((np.ones(num_holding), np.zeros(num_holding))))
        A_raw = matrix(np.ones(num_holding), (1,num_holding))
        b_raw = matrix(1.0)

        # prepare
        benchmark_stock_list = self.benchmark_index_list_dict[date]

        X_style = pd.DataFrame()
        for class_name in self.class_factor_dict_adj.keys():
            X_style[class_name] = self.class_factor_dict_adj[class_name][date]

        X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: 改进处理nan的方法
        if not self.use_dynamic_ind:
            X_ind_holding = self.ind_df.T[holding_stock_list]
        else:
            X_ind_holding = self.dynamic_ind_data_ohe(date, holding_stock_list).T
        X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
        benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]
        benchmark_stock_ind = self.dynamic_ind_data_ohe(date, benchmark_stock_list).T

        # start loop
        status = False
        fail_to_constrain = False
        while not status:  
            # deep copy 
            G = matrix(G_raw)
            h = matrix(h_raw)
            A = matrix(A_raw)
            b = matrix(b_raw)
            P = matrix(P_raw)
            q = matrix(q_raw)

            # 风格中性
            if cfg.style_neutralize:
                benchmark_current_style = np.matmul(X_style_benchmark, benchmark_current_weight).values  # 基准指数风格
                style_num = X_style.shape[1]
                # style_low_limit = cfg.style_low_limit
                # style_high_limit = cfg.style_high_limit   # TODO: 不同风格因子值不同，是否每个单独设置暴露
                style_low = np.matrix(np.full(style_num, style_low_limit) + benchmark_current_style).T
                style_high = np.matrix(np.full(style_num, style_high_limit) + benchmark_current_style).T

                G = matrix(np.vstack((G,X_style_holding.values, -X_style_holding.values)))
                h = matrix(np.vstack((h, style_high,-style_low)))

            # 行业中性
            if cfg.ind_neutralize:
                if self.use_dynamic_ind:
                    benchmark_current_ind = np.matmul(benchmark_stock_ind, benchmark_current_weight).values
                    ind_num = self.num_dyn_ind
                else: 
                    benchmark_current_ind = np.matmul(self.ind_df.T[benchmark_stock_list], benchmark_current_weight).values
                    ind_num = self.ind_df.shape[1]
                # ind_low_limit = cfg.ind_low_limit
                # ind_high_limit = cfg.ind_high_limit
                ind_low = np.matrix(np.full(ind_num, ind_low_limit) + benchmark_current_ind).T
                ind_high = np.matrix(np.full(ind_num, ind_high_limit) + benchmark_current_ind).T
                G = matrix(np.vstack((G, X_ind_holding.values, -X_ind_holding.values)))
                h = matrix(np.vstack((h, ind_high,-ind_low)))

            # 限制换手率
            
            if cfg.turnover_constraint:

                previous_day_signal = prev_holdings # opt_signal.shift(1, axis = 1).fillna(0)[date][holding_stock_list].values
                fixed_turnover = 1 - previous_day_signal.sum()

                # create n extra variables
                objective_turnover = turnover_limit - fixed_turnover

                P = matrix(np.hstack((np.vstack((P,np.zeros((num_holding, num_holding)))), np.zeros((2 *num_holding, num_holding)))))
                q = matrix(np.vstack((q, np.zeros( (num_holding, 1)))))
                identity_holding = np.identity(num_holding)
                G = matrix(np.vstack((np.hstack((G, np.zeros((G.size[0], num_holding)))), np.block([[identity_holding, -identity_holding], [-identity_holding, -identity_holding]]))))
                h = matrix(np.vstack((h, previous_day_signal.reshape(num_holding,1), -previous_day_signal.reshape(num_holding,1))))
                A = matrix(np.vstack((np.hstack((A, np.zeros((1, num_holding)))),np.hstack((np.zeros((1, num_holding)), np.ones((1, num_holding)))))))
                b = matrix(np.vstack((b, matrix(objective_turnover))))

            opts = {'maxiters' : 200, 'show_progress': False}
            
            try:
                sol=solvers.qp(P, q, G, h, A, b, options = opts)
                solution = np.array(sol['x'])
                status = 'optimal' in sol['status']
            except:
                traceback_msg = traceback.format_exc()
                status = False

            if not status:

                # !!! 无法converge的时候调整，先调整风格暴露约束，然后调整行业约束，最后调整换手率限制

                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0:

                        if turnover_limit >= 2.0 :
                            # 所有约束失败
                            status = True
                            fail_to_constrain = True
                            print(traceback_msg)
                            
                        else:
                            style_low_limit = self.all_style_low_limit
                            style_high_limit = self.all_style_high_limit
                            ind_high_limit = self.all_ind_high_limit
                            ind_low_limit = self.all_ind_low_limit
                            turnover_limit = turnover_limit + 0.1
                    
                    else:
                        style_low_limit = self.all_style_low_limit
                        style_high_limit = self.all_style_high_limit
                        
                        ind_high_limit = ind_high_limit + 0.05
                        ind_low_limit = ind_low_limit - 0.05


                else:
                    style_low_limit = style_low_limit - 0.1
                    style_high_limit = style_high_limit + 0.1

        weight = np.array([w[0] for w in solution])[:num_holding]

        # adjust weight (prevent negative weights when fail to constrain)
        if fail_to_constrain:
            logging.warn(f'{date} fail to constrain: discard negative weights')
            weight[weight < 0] = 0 
            weight = weight / weight.sum()  # re-normalize the rest 

        return weight



    def qp_method_2(self, date, Sigma_holding_stock, opt_signal):
        num_holding = Sigma_holding_stock.shape[0]
        holding_stock_list = self.holding_stock_list_dict[date]

        weight_low = self.weight_low
        weight_high = self.weight_high
        # turnover_limit = self.all_turnover_limit
        style_low_limit = self.all_style_low_limit
        style_high_limit = self.all_style_high_limit   
        ind_low_limit = self.all_ind_low_limit
        ind_high_limit = self.all_ind_high_limit
        previous_day_signal = opt_signal.shift(1, axis = 1).fillna(0)[date][holding_stock_list].values

        
        status = False
        while not status:

            if 'ret_var' in self.obj_func:  # 最大化经风险调整后收益
                P = 2* matrix(Sigma_holding_stock)

                # 预测收益率
                current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0) # TODO: 改进处理nan的方法
            else:
                print('Cannot support this objective function')
                break

            # 约束条件：持仓权重上下限
            G = matrix(np.vstack((np.identity(num_holding), -np.identity(num_holding))))
            h = matrix(np.hstack((np.full(num_holding, weight_high), np.full(num_holding, -weight_low))))
            #h = matrix(np.hstack((np.ones(num_holding), np.zeros(num_holding))))
            A = matrix(np.ones(num_holding), (1,num_holding))
            b = matrix(1.0)
            
            # 约束条件：行业风格中性
            
            benchmark_stock_list = self.benchmark_index_list_dict[date]

            X_style = pd.DataFrame()
            for class_name in self.class_factor_dict_adj.keys():
                X_style[class_name] = self.class_factor_dict_adj[class_name][date]

            X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: 改进处理nan的方法
            X_ind_holding = self.ind_df.T[holding_stock_list]
            X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
            benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]

            # 风格中性
            if cfg.style_neutralize:
                benchmark_current_style = np.matmul(X_style_benchmark, benchmark_current_weight).values  # 基准指数风格
                style_num = X_style.shape[1]
                # style_low_limit = cfg.style_low_limit
                # style_high_limit = cfg.style_high_limit   # TODO: 不同风格因子值不同，是否每个单独设置暴露
                style_low = np.matrix(np.full(style_num, style_low_limit) + benchmark_current_style).T
                style_high = np.matrix(np.full(style_num, style_high_limit) + benchmark_current_style).T

                G = matrix(np.vstack((G,X_style_holding.values, -X_style_holding.values)))
                h = matrix(np.vstack((h, style_high,-style_low)))

            # 行业中性
            if cfg.ind_neutralize:
                benchmark_current_ind = np.matmul(self.ind_df.T[benchmark_stock_list], benchmark_current_weight).values
                ind_num = self.ind_df.shape[1]
                # ind_low_limit = cfg.ind_low_limit
                # ind_high_limit = cfg.ind_high_limit
                ind_low = np.matrix(np.full(ind_num, ind_low_limit) + benchmark_current_ind).T
                ind_high = np.matrix(np.full(ind_num, ind_high_limit) + benchmark_current_ind).T
                G = matrix(np.vstack((G,X_ind_holding.values, -X_ind_holding.values)))
                h = matrix(np.vstack((h, ind_high,-ind_low)))

            P = matrix(2 * self.penalty_lambda* np.block([[Sigma_holding_stock, -Sigma_holding_stock], [-Sigma_holding_stock, Sigma_holding_stock]]))

            q = matrix(np.hstack((2 * self.penalty_lambda * Sigma_holding_stock @ previous_day_signal + np.full(num_holding, self.penalty_theta) - current_holding_pred_return.values,
            -2 * self.penalty_lambda * Sigma_holding_stock @ previous_day_signal + np.full(num_holding, self.penalty_theta) + current_holding_pred_return.values)))

            G_new = matrix(np.vstack((np.hstack((G, -G)), -np.identity(2 * num_holding) )))

            h = matrix(np.vstack(((np.array([(np.array([w.squeeze() for w in np.array(h)]) - (G @ previous_day_signal))]).T, np.zeros((2 * num_holding, 1))))) )

            # G = matrix(np.vstack((np.hstack((G, -G)), -np.identity(2 * num_holding) )))

            b = matrix((np.array([w.squeeze() for w in np.array(b)]) - (A @ previous_day_signal)))

            A = matrix(np.hstack((A, -A)))

            opts = {'maxiters' : 200, 'show_progress': False}

            try:
                sol=solvers.qp(P, q, G_new, h, A, b, options = opts)
                # solution = np.array(sol['x'][:num_holding]) - np.array(sol['x'][num_holding:])
                status = 'optimal' in sol['status']
            except:
                status = False

            if not status:

                # 无法converge的时候调整，先调整风格暴露约束，再调整行业约束
                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0 :
                        # 所有约束失败
                        # TODO: sol没打印出来
                        status = True
                        print(date)
                        print('fail to constraint')
                    else:
                        style_low_limit = self.all_style_low_limit
                        style_high_limit = self.all_style_high_limit
                        ind_high_limit = ind_high_limit + 0.05
                        ind_low_limit = ind_low_limit - 0.05

                else:
                    style_low_limit = style_low_limit - 0.1
                    style_high_limit = style_high_limit + 0.1
                
            # sol=solvers.qp(P, q, G_new, h, A, b, options = opts)
        solution = np.array(sol['x'][:num_holding]) - np.array(sol['x'][num_holding:]) 

        weight = np.array([w.squeeze() for w in solution]) + previous_day_signal

        return weight


    def qp_method_3(self, date, Sigma_holding_stock, opt_signal):
        """
        模式三: 风格暴露成为惩罚项
        """
        num_holding = Sigma_holding_stock.shape[0]
        holding_stock_list = self.holding_stock_list_dict[date]

        weight_low = self.weight_low
        weight_high = self.weight_high
        style_low_limit = self.all_style_low_limit
        style_high_limit = self.all_style_high_limit   
        ind_low_limit = self.all_ind_low_limit
        ind_high_limit = self.all_ind_high_limit
        previous_day_signal = opt_signal.shift(1, axis=1).fillna(0)[date][holding_stock_list].values
        today_corresponding_benchmark_weight = self.benchmark_index_weight_filled[date][holding_stock_list].values  # 今日对应的成分股部分
        status = False
        while not status:

            # TODO: 其他objective function
            if 'ret_var' in self.obj_func:  # 最大化经风险调整后收益
                P = 2 * matrix(Sigma_holding_stock)

                # 预测收益率
                current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0) # TODO: 改进处理nan的方法
                # q = matrix(-current_holding_pred_return / self.penalty_lambda)
            else:
                print('Cannot support this objective function')
                raise NotImplementedError
                break

            
            # 约束条件：持仓权重上下限
            G = matrix(np.vstack((np.identity(num_holding), -np.identity(num_holding))))
            h = matrix(np.hstack((np.full(num_holding, weight_high), np.full(num_holding, -weight_low))))
            A = matrix(np.ones(num_holding), (1,num_holding))
            b = matrix(1.0)
            
            # 约束条件：行业风格中性
            benchmark_stock_list = self.benchmark_index_list_dict[date]

            X_style = pd.DataFrame()
            for class_name in self.class_factor_dict_adj.keys():
                X_style[class_name] = self.class_factor_dict_adj[class_name][date]

            X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: 改进处理nan的方法
            X_style_holding_quad = X_style_holding.T @ X_style_holding 
            X_ind_holding = self.ind_df.T[holding_stock_list]
            # X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
            previous_holding_excess = previous_day_signal - today_corresponding_benchmark_weight  # 昨日持仓部分的超额权重
            
            benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]

            # 行业中性
            if cfg.ind_neutralize:
                benchmark_current_ind = np.matmul(self.ind_df.T[benchmark_stock_list], benchmark_current_weight).values
                ind_num = self.ind_df.shape[1]
                ind_low = np.matrix(np.full(ind_num, ind_low_limit) + benchmark_current_ind).T
                ind_high = np.matrix(np.full(ind_num, ind_high_limit) + benchmark_current_ind).T
                G = matrix(np.vstack((G,X_ind_holding.values, -X_ind_holding.values)))
                h = matrix(np.vstack((h, ind_high,-ind_low)))


            P = matrix(2 * self.penalty_lambda* 
                    np.block([
                        [Sigma_holding_stock, - Sigma_holding_stock], 
                        [- Sigma_holding_stock, Sigma_holding_stock]
                    ]) + 
                    2 * self.penalty_nu* 
                    np.block([
                        [X_style_holding_quad, - X_style_holding_quad], 
                        [- X_style_holding_quad, X_style_holding_quad]
                    ])
                )
            q = matrix(np.hstack(
                (2 * self.penalty_lambda * Sigma_holding_stock @ previous_day_signal + 2 * self.penalty_nu * X_style_holding_quad @ previous_holding_excess +  \
                 np.full(num_holding, self.penalty_theta) - current_holding_pred_return.values,
               - 2 * self.penalty_lambda * Sigma_holding_stock @ previous_day_signal - 2 * self.penalty_nu * X_style_holding_quad @ previous_holding_excess + \
                 np.full(num_holding, self.penalty_theta) + current_holding_pred_return.values)))

            G_new = matrix(np.vstack((np.hstack((G, -G)), -np.identity(2 * num_holding) )))

            h = matrix(np.vstack(((np.array([(np.array([w.squeeze() for w in np.array(h)]) - (G @ previous_day_signal))]).T, np.zeros((2 * num_holding, 1))))) )

            b = matrix((np.array([w.squeeze() for w in np.array(b)]) - (A @ previous_day_signal)))

            A = matrix(np.hstack((A, -A)))

            opts = {'maxiters' : 200, 'show_progress': False}

            try:
                sol = solvers.qp(P, q, G_new, h, A, b, options=opts)
                status = 'optimal' in sol['status']
            except:
                status = False

            if not status:
                # 无法converge的时候调整，先调整风格暴露约束，再调整行业约束
                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0 :
                        # 所有约束失败
                        status = True
                        print(date)
                        print('fail to constraint')
                    else:
                        style_low_limit = self.all_style_low_limit
                        style_high_limit = self.all_style_high_limit
                        ind_high_limit = ind_high_limit + 0.05
                        ind_low_limit = ind_low_limit - 0.05

                else:
                    style_low_limit = style_low_limit - 0.1
                    style_high_limit = style_high_limit + 0.1

        solution = np.array(sol['x'][:num_holding]) - np.array(sol['x'][num_holding:]) 
        weight = np.array([w.squeeze() for w in solution]) + previous_day_signal

        return weight

    def qp_method_4(self, date, Sigma_holding_stock, opt_signal):
        """
        模式四: 模型错位, 惩罚未被风格因子解释掉的收益
        """
        num_holding = Sigma_holding_stock.shape[0]
        holding_stock_list = self.holding_stock_list_dict[date]

        weight_low = self.weight_low
        weight_high = self.weight_high
        turnover_limit = self.all_turnover_limit
        style_low_limit = self.all_style_low_limit
        style_high_limit = self.all_style_high_limit   
        ind_low_limit = self.all_ind_low_limit
        ind_high_limit = self.all_ind_high_limit
        previous_day_signal = opt_signal.shift(1, axis = 1).fillna(0)[date][holding_stock_list].values

        status = False
        while not status:
            
            benchmark_stock_list = self.benchmark_index_list_dict[date]

            X_style = pd.DataFrame()
            for class_name in self.class_factor_dict_adj.keys():
                X_style[class_name] = self.class_factor_dict_adj[class_name][date]

            X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: 改进处理nan的方法
            X_style_holding_np = X_style_holding.values
            X_ind_holding = self.ind_df.T[holding_stock_list]
            X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
            benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]
            
            
            # 目标函数
            if 'min_var' in self.obj_func:
                
                P = 2 * matrix(Sigma_holding_stock) 
                q = matrix(np.zeros(num_holding))

            elif 'ret_var' in self.obj_func:  # 最大化经风险调整后收益
                # 预测收益率
                current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0).values # TODO: 改进处理nan的方法
                style_explained_pred_return = X_style_holding_np.T @ np.linalg.inv(X_style_holding_np @ X_style_holding_np.T) @ X_style_holding_np @ current_holding_pred_return
                style_unexplained_pred_return = current_holding_pred_return - style_explained_pred_return
                style_unexplained_pred_return = np.array([style_unexplained_pred_return]).T  # 增广, 变成column vector
                q = matrix(- current_holding_pred_return)

                P = matrix(
                    2 * Sigma_holding_stock * self.penalty_lambda \
                  + 2 * style_unexplained_pred_return @ style_unexplained_pred_return.T * self.penalty_xi
                )
            else:
                raise NotImplementedError
                print('Cannot support this objective function')

            # 约束条件：持仓权重上下限
            G = matrix(np.vstack((np.identity(num_holding), -np.identity(num_holding))))
            h = matrix(np.hstack((np.full(num_holding, weight_high), np.full(num_holding, -weight_low))))
            A = matrix(np.ones(num_holding), (1,num_holding))
            b = matrix(1.0)
            
            # 约束条件
            # 风格中性
            if cfg.style_neutralize:
                benchmark_current_style = np.matmul(X_style_benchmark, benchmark_current_weight).values  # 基准指数风格
                style_num = X_style.shape[1]
                style_low = np.matrix(np.full(style_num, style_low_limit) + benchmark_current_style).T
                style_high = np.matrix(np.full(style_num, style_high_limit) + benchmark_current_style).T
                G = matrix(np.vstack((G,X_style_holding.values, -X_style_holding.values)))
                h = matrix(np.vstack((h, style_high,-style_low)))

            # 行业中性
            if cfg.ind_neutralize:
                benchmark_current_ind = np.matmul(self.ind_df.T[benchmark_stock_list], benchmark_current_weight).values
                ind_num = self.ind_df.shape[1]
                ind_low = np.matrix(np.full(ind_num, ind_low_limit) + benchmark_current_ind).T
                ind_high = np.matrix(np.full(ind_num, ind_high_limit) + benchmark_current_ind).T
                G = matrix(np.vstack((G,X_ind_holding.values, -X_ind_holding.values)))
                h = matrix(np.vstack((h, ind_high,-ind_low)))

            # 限制换手率
            if cfg.turnover_constraint:
                previous_day_signal = opt_signal.shift(1, axis = 1).fillna(0)[date][holding_stock_list].values
                fixed_turnover = 1 - previous_day_signal.sum()

                # create n extra variables
                objective_turnover = turnover_limit - fixed_turnover

                P = matrix(np.hstack((np.vstack((P,np.zeros((num_holding, num_holding)))), np.zeros((2 *num_holding, num_holding)))))
                q = matrix(np.vstack((q, np.zeros( (num_holding, 1)))))
                identity_holding = np.identity(num_holding)
                G = matrix(np.vstack((np.hstack((G, np.zeros((G.size[0], num_holding)))), np.block([[identity_holding, -identity_holding], [-identity_holding, -identity_holding]]))))
                h = matrix(np.vstack((h, previous_day_signal.reshape(num_holding,1), -previous_day_signal.reshape(num_holding,1))))
                A = matrix(np.vstack((np.hstack((A, np.zeros((1, num_holding)))),np.hstack((np.zeros((1, num_holding)), np.ones((1, num_holding)))))))
                b = matrix(np.vstack((b, matrix(objective_turnover))))

            opts = {'maxiters' : 200, 'show_progress': False}
            
            try:
                sol=solvers.qp(P, q, G, h, A, b, options = opts)
                solution = np.array(sol['x'])
                status = 'optimal' in sol['status']
            except:
                status = False

            if not status:

                # !!! 无法converge的时候调整，先调整风格暴露约束，然后调整行业约束，最后调整换手率限制

                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0:

                        if turnover_limit >= 2.0 :
                            # 所有约束失败
                            status = True
                            print(date)
                            print('fail to constraint')
                        else:
                            style_low_limit = self.all_style_low_limit
                            style_high_limit = self.all_style_high_limit
                            ind_high_limit = self.all_ind_high_limit
                            ind_low_limit = self.all_ind_low_limit
                            turnover_limit = turnover_limit + 0.1
                    
                    else:
                        style_low_limit = self.all_style_low_limit
                        style_high_limit = self.all_style_high_limit
                        
                        ind_high_limit = ind_high_limit + 0.05
                        ind_low_limit = ind_low_limit - 0.05
                
                else:
                    style_low_limit = style_low_limit - 0.1
                    style_high_limit = style_high_limit + 0.1


        weight = np.array([w[0] for w in solution])[:num_holding]



        return weight
    
    def cal_opt_weight(self,return_type):
        '''最佳持仓权重计算'''
        

        sigma_holding = self.sigma_holding_dict[return_type]

        opt_signal = pd.DataFrame(0, columns=self.date_list_opt, index=self.tickers)
        self.obj_func = cfg.obj_func
        # weight_low = self.weight_low
        # weight_high = self.weight_high
        
        # 读入预测收益率
        self.pred_ret_df = self.read_ml_factor_data(cfg.ml_factor_name, self.tickers, self.date_list)
        # fail_date_list =[]

        # 选择优化模型
        qp_method_func = None 
        if self.qp_method == 1:
            qp_method_func = self.qp_method_1
        elif self.qp_method == 2:
            qp_method_func = self.qp_method_2 
        elif self.qp_method == 3:
            qp_method_func = self.qp_method_3 
        elif self.qp_method == 4: 
            qp_method_func = self.qp_method_4
        else:
            raise NotImplementedError

        # TODO: simplify codes

        # set up template to record previous holdings 
        template_series = pd.Series(0, index=self.tickers)
        prev_holdings = template_series.copy()

        # print dates 
        prev_year_date = self.date_list_opt[0][:6]
        start = time.time()

        # 开始循环
        for date in self.date_list_opt:
            # extract today data
            Sigma_holding_stock = sigma_holding[date]
            today_holdings = self.holding_stock_list_dict[date]
            prev_holdings_today = prev_holdings[today_holdings].fillna(0)
            # print(date, '\n\n', Sigma_holding_stock, '\n\n', prev_holdings)
            weight = qp_method_func(date, Sigma_holding_stock, prev_holdings_today.values)

            # append to weight
            opt_signal[date][today_holdings] = weight

            # update prev_holdings
            prev_holdings = pd.Series(weight, index=today_holdings)
            prev_holdings = prev_holdings + template_series

            # print by month
            cur_year_date = date[:6]
            if cur_year_date != prev_year_date:
                # log 
                print(f'{prev_year_date} takes {time.time() - start:.2f} s')
                # renew 
                prev_year_date = cur_year_date
                start = time.time() 
        self.opt_signal_dict[return_type] = opt_signal



    def save_opt_signal(self):
        """
        储存优化权重
        """
        output_signal_path = cfg.output_signal_path
        for return_type in self.return_type_list:
            if not os.path.isdir(output_signal_path):
                os.mkdir(output_signal_path)

            file_name = '{}/{}{}{}_{}_{}_{}'.format(
                output_signal_path,
                cfg.input_signal_df_name,  
                cfg.adj_method,
                cfg.weight_low, 
                cfg.weight_high, 
                return_type,
                self.obj_func,
            )
            if 'ret_var' in self.obj_func:
                file_name = file_name + f'_penalty{self.penalty_lambda}'
            file_name = file_name + f'_method{self.qp_method}'
            if self.qp_method == 1: 
                file_name = (
                    file_name + 
                    cfg.style_neutralize * f'_style{cfg.style_high_limit}' + 
                    cfg.ind_neutralize * f'_ind{cfg.ind_high_limit}' + 
                    cfg.turnover_constraint * f'_turnover{cfg.turnover_limit}'
                )
            elif self.qp_method == 2:
                file_name = (
                    file_name + 
                    f'_cost_penalty{self.penalty_theta}' + 
                    cfg.style_neutralize * f'_style{cfg.style_high_limit}' + 
                    cfg.ind_neutralize * f'_ind{cfg.ind_high_limit}'
                )
            elif self.qp_method == 3:
                file_name = (
                    file_name + 
                    f'_style_penalty{self.penalty_nu}' + 
                    cfg.ind_neutralize * f'_ind{cfg.ind_high_limit}'
                )
            elif self.qp_method == 4: 
                file_name = (
                    file_name + 
                    f'_misalign_penalty{self.penalty_xi}' + 
                    cfg.style_neutralize * f'_style{cfg.style_high_limit}' + 
                    cfg.ind_neutralize * f'_ind{cfg.ind_high_limit}' + 
                    cfg.turnover_constraint * f'_turnover{cfg.turnover_limit}'
                )
            if self.use_dynamic_ind:
                file_name += self.dynamic_ind_name
            # self.opt_signal_dict[return_type].to_csv(file_name + '.csv')
            # self.myconnector.save_eod_feature(
            #     file_name, self.opt_signal_dict[return_type],
            # )
            self.opt_signal_dict[return_type].reset_index().to_feather(file_name)
            print(file_name)

            # TODO: 输出文件名太长了，是否输出编号，然后自己记录参数？


    def start_weight_optimize_process(self):
        print("正在读取数据")
        t0 = time.time()
        self.prepare_data()
        print("读取数据耗时", time.time() - t0)

        print("正在计算持仓权重")
        t0 = time.time()
        for ret_type in self.return_type_list:
            self.cal_sigma_holding(ret_type)
            self.cal_opt_weight(ret_type)

        print("计算持仓权重耗时", time.time() - t0)
        
        # 数据结果储存
        print("正在储存数据")
        t0 = time.time()
        self.save_opt_signal()
        print("储存数据耗时", time.time() - t0)


