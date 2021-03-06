"""
ζιδΌε
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
        
        # to be populated 
        self.factor_cov_dict = {}
        self.idio_var_dict = {}
        self.cov_save_path = cfg.cov_save_path
        self.return_type_list = cfg.return_type_list

        self.class_factor_dict_adj = {}

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

    def load_history(self):
        # dataserver
        self.myconnector = PqiDataSdkOffline()

        # stock pools and dates
        self.all_stocks = self.myconnector.get_ticker_list()
        self.eod_data_dict = self.myconnector.get_eod_history(
            tickers=self.all_stocks, 
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.trade_dates = self.myconnector.select_trade_dates(
            start_date=self.start_date, 
            end_date=self.end_date
        )
        self.tickers = list(self.eod_data_dict["ClosePrice"].index)  
        self.date_list = list(self.eod_data_dict['ClosePrice'].columns)
    
    # ==============================================
    # -------------- data prep ---------------------
    # ==============================================

    def read_cov_data(self, return_type):
        """ 
        read factor cov and idio cov
        θ―»εε ε­εηΉθ΄¨ζΆηηεζΉε·?η©ι΅ 
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
        θ―»εζζεζΉε·?δΌ°θ?‘ 
        """
        for name in self.return_type_list:
            self.factor_cov_dict[name], self.idio_var_dict[name] = self.read_cov_data(name)


    def load_signal_data(self):
        self.input_signal = pd.read_feather(
            os.path.join(cfg.input_signal_path, cfg.input_signal_df_name)
        ).set_index('index')


    def read_factor_data(self,feature_name, tickers, date_list):
        """ θ―»εε ε­ """
        
        # path = cfg.class_factor_path
        
        # feature_name = "eod_" + feature_name
        # factor = self.myconnector.get_eod_feature(fields=feature_name, where=path, tickers=tickers, dates=date_list)
        factor_df = self.myconnector.read_eod_feature(
            feature_name, des='risk_factor/class_factors', dates=date_list
        )
        factor_df = factor_df.shift(-1, axis=1)     # * θ½¬ζ’εδ»ζ₯ηι£ζ Όε ε­εΌ
        # return factor[feature_name].to_dataframe().shift(-1, axis = 1)  
        return factor_df


    def load_factor_data(self):
        for name in self.class_name:
            self.class_factor_dict_adj[name] = self.read_factor_data(name, self.tickers , self.date_list)

    # -------- static ind ---------- 
    def get_ind_date(self):
        """ read industry data """
        # θ―»εθ‘δΈζ°ζ?
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
        sample_cross_section = self.ind_df.iloc[:, -1]
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
        """ read ml predicted returns (factor) θ―»εmlι’ζ΅ζΆηηε ε­ζ°ζ? """
        # θ―»εεε§ε ε­εΌ
        factor_df = self.myconnector.read_eod_feature(
            feature_name, des='ml_factor', dates=date_list
        )
        return factor_df

    def get_opt_dates(self):
        """
        get date list (date intersection of signals, cov estimates, and basic data)
        θ·ειθ¦η»εζιδΌεηζ₯ζεθ‘¨οΌζδ»δΏ‘ε·εεζΉε·?η©ι΅ηζ₯ζδΊ€ιοΌ
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
        θ―»εζζ°ζεθ‘ζιοΌθ½¬ζ’ζζζ°ζεθ‘mask: ε½ε€©ζθ‘η₯¨δΈΊθ―₯ζζ°ζεθ‘δΈΊ1οΌε¦εδΈΊnan
        :param index_list: δΈδΈͺθ£ζζζ°ηεθ‘¨οΌζ―ζη»εζζ°
        :return ζζ°mask
        """
        agg_index_mask = False
        # θ―»ε₯εζΌζ₯ε€δΈͺindex
        for index in index_list:
            index_weight = self.get_stock_weight(index)
            index_mask = index_weight.notna()
            agg_index_mask = agg_index_mask | index_mask
        # ε°dataframeθ½¬ζ’δΈΊ1εnanηη©ι΅
        agg_index_mask = agg_index_mask.astype(int) / agg_index_mask.astype(int)
        return agg_index_mask   

    # =====================================================
    # ------------------ compute sigma --------------------
    # =====================================================

    def cal_sigma_holding(self, return_type):
        """
        compute daily holding sigma 
        θ?‘η?ζ―ζ₯ζδ»θ‘η₯¨ηεζΉε·?η©ι΅Sigma
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
            
            # ε½ζ₯ζδ»θ‘η₯¨
            holding_stock_list = [x.zfill(6) for x in [str(x) for x in list(current_date_input_signal[current_date_input_signal > 1e-6].index)]]  # !!!
            self.holding_stock_list_dict[date] = holding_stock_list

            # ε½ζ₯ζζ°ζεθ‘
            self.benchmark_index_list_dict[date] = self.benchmark_index_weight[self.benchmark_index_weight[date].notna()].index
            
            # factor cov and idio cov 
            Sigma_F = factor_cov_est[date]
            diag_E = idio_var_est[date][holding_stock_list].values[0]
            diag_E[np.isnan(diag_E)] = np.nanmean(diag_E)  # TODO: ζΉθΏε€ηnanηζΉζ³
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

            X_holding = X.T[holding_stock_list].fillna(0)  # TODO: ζΉθΏε€ηnanηζΉζ³
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

        elif 'ret_var' in self.obj_func:  # ζε€§εη»ι£ι©θ°ζ΄εζΆη
            P_raw = 2 * matrix(Sigma_holding_stock)

            # ι’ζ΅ζΆηη
            current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0) # TODO: ζΉθΏε€ηnanηζΉζ³
            q_raw = matrix(-current_holding_pred_return / self.penalty_lambda)
        else:
            raise NotImplementedError('Cannot support this objective function')

        # TODO: sparse matrix
        # ηΊ¦ζζ‘δ»ΆοΌζδ»ζιδΈδΈι base quadratic functions 
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

        X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: ζΉθΏε€ηnanηζΉζ³
        if not self.use_dynamic_ind:
            X_ind_holding = self.ind_df.T[holding_stock_list]
        else:
            X_ind_holding = self.dynamic_ind_data_ohe(date, holding_stock_list).T
            benchmark_stock_ind = self.dynamic_ind_data_ohe(date, benchmark_stock_list).T
        X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
        benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]

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

            # ι£ζ ΌδΈ­ζ§
            if cfg.style_neutralize:
                benchmark_current_style = np.matmul(X_style_benchmark, benchmark_current_weight).values  # εΊεζζ°ι£ζ Ό
                style_num = X_style.shape[1]
                # style_low_limit = cfg.style_low_limit
                # style_high_limit = cfg.style_high_limit   # TODO: δΈει£ζ Όε ε­εΌδΈεοΌζ―ε¦ζ―δΈͺεη¬θ?Ύη½?ζ΄ι²
                style_low = np.matrix(np.full(style_num, style_low_limit) + benchmark_current_style).T
                style_high = np.matrix(np.full(style_num, style_high_limit) + benchmark_current_style).T

                G = matrix(np.vstack((G,X_style_holding.values, -X_style_holding.values)))
                h = matrix(np.vstack((h, style_high,-style_low)))

            # θ‘δΈδΈ­ζ§
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

            # ιεΆζ’ζη
            
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

                # !!! ζ ζ³convergeηζΆεθ°ζ΄οΌεθ°ζ΄ι£ζ Όζ΄ι²ηΊ¦ζοΌηΆεθ°ζ΄θ‘δΈηΊ¦ζοΌζεθ°ζ΄ζ’ζηιεΆ

                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0:

                        if turnover_limit >= 2.0 :
                            # ζζηΊ¦ζε€±θ΄₯
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

            if 'ret_var' in self.obj_func:  # ζε€§εη»ι£ι©θ°ζ΄εζΆη
                P = 2* matrix(Sigma_holding_stock)

                # ι’ζ΅ζΆηη
                current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0) # TODO: ζΉθΏε€ηnanηζΉζ³
            else:
                print('Cannot support this objective function')
                break

            # ηΊ¦ζζ‘δ»ΆοΌζδ»ζιδΈδΈι
            G = matrix(np.vstack((np.identity(num_holding), -np.identity(num_holding))))
            h = matrix(np.hstack((np.full(num_holding, weight_high), np.full(num_holding, -weight_low))))
            #h = matrix(np.hstack((np.ones(num_holding), np.zeros(num_holding))))
            A = matrix(np.ones(num_holding), (1,num_holding))
            b = matrix(1.0)
            
            # ηΊ¦ζζ‘δ»ΆοΌθ‘δΈι£ζ ΌδΈ­ζ§
            
            benchmark_stock_list = self.benchmark_index_list_dict[date]

            X_style = pd.DataFrame()
            for class_name in self.class_factor_dict_adj.keys():
                X_style[class_name] = self.class_factor_dict_adj[class_name][date]

            X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: ζΉθΏε€ηnanηζΉζ³
            X_ind_holding = self.ind_df.T[holding_stock_list]
            X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
            benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]

            # ι£ζ ΌδΈ­ζ§
            if cfg.style_neutralize:
                benchmark_current_style = np.matmul(X_style_benchmark, benchmark_current_weight).values  # εΊεζζ°ι£ζ Ό
                style_num = X_style.shape[1]
                # style_low_limit = cfg.style_low_limit
                # style_high_limit = cfg.style_high_limit   # TODO: δΈει£ζ Όε ε­εΌδΈεοΌζ―ε¦ζ―δΈͺεη¬θ?Ύη½?ζ΄ι²
                style_low = np.matrix(np.full(style_num, style_low_limit) + benchmark_current_style).T
                style_high = np.matrix(np.full(style_num, style_high_limit) + benchmark_current_style).T

                G = matrix(np.vstack((G,X_style_holding.values, -X_style_holding.values)))
                h = matrix(np.vstack((h, style_high,-style_low)))

            # θ‘δΈδΈ­ζ§
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

                # ζ ζ³convergeηζΆεθ°ζ΄οΌεθ°ζ΄ι£ζ Όζ΄ι²ηΊ¦ζοΌεθ°ζ΄θ‘δΈηΊ¦ζ
                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0 :
                        # ζζηΊ¦ζε€±θ΄₯
                        # TODO: solζ²‘ζε°εΊζ₯
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
        ζ¨‘εΌδΈ: ι£ζ Όζ΄ι²ζδΈΊζ©η½ι‘Ή
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
        today_corresponding_benchmark_weight = self.benchmark_index_weight_filled[date][holding_stock_list].values  # δ»ζ₯ε―ΉεΊηζεθ‘ι¨ε
        status = False
        while not status:

            # TODO: εΆδ»objective function
            if 'ret_var' in self.obj_func:  # ζε€§εη»ι£ι©θ°ζ΄εζΆη
                P = 2 * matrix(Sigma_holding_stock)

                # ι’ζ΅ζΆηη
                current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0) # TODO: ζΉθΏε€ηnanηζΉζ³
                # q = matrix(-current_holding_pred_return / self.penalty_lambda)
            else:
                print('Cannot support this objective function')
                raise NotImplementedError
                break

            
            # ηΊ¦ζζ‘δ»ΆοΌζδ»ζιδΈδΈι
            G = matrix(np.vstack((np.identity(num_holding), -np.identity(num_holding))))
            h = matrix(np.hstack((np.full(num_holding, weight_high), np.full(num_holding, -weight_low))))
            A = matrix(np.ones(num_holding), (1,num_holding))
            b = matrix(1.0)
            
            # ηΊ¦ζζ‘δ»ΆοΌθ‘δΈι£ζ ΌδΈ­ζ§
            benchmark_stock_list = self.benchmark_index_list_dict[date]

            X_style = pd.DataFrame()
            for class_name in self.class_factor_dict_adj.keys():
                X_style[class_name] = self.class_factor_dict_adj[class_name][date]

            X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: ζΉθΏε€ηnanηζΉζ³
            X_style_holding_quad = X_style_holding.T @ X_style_holding 
            X_ind_holding = self.ind_df.T[holding_stock_list]
            # X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
            previous_holding_excess = previous_day_signal - today_corresponding_benchmark_weight  # ζ¨ζ₯ζδ»ι¨εηθΆι’ζι
            
            benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]

            # θ‘δΈδΈ­ζ§
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
                # ζ ζ³convergeηζΆεθ°ζ΄οΌεθ°ζ΄ι£ζ Όζ΄ι²ηΊ¦ζοΌεθ°ζ΄θ‘δΈηΊ¦ζ
                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0 :
                        # ζζηΊ¦ζε€±θ΄₯
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
        ζ¨‘εΌε: ζ¨‘ειδ½, ζ©η½ζͺθ’«ι£ζ Όε ε­θ§£ιζηζΆη
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

            X_style_holding = X_style.T[holding_stock_list].fillna(0) # TODO: ζΉθΏε€ηnanηζΉζ³
            X_style_holding_np = X_style_holding.values
            X_ind_holding = self.ind_df.T[holding_stock_list]
            X_style_benchmark = X_style.T[benchmark_stock_list].fillna(0)
            benchmark_current_weight = self.benchmark_index_weight[date][self.benchmark_index_weight[date].notna()]
            
            
            # η?ζ ε½ζ°
            if 'min_var' in self.obj_func:
                
                P = 2 * matrix(Sigma_holding_stock) 
                q = matrix(np.zeros(num_holding))

            elif 'ret_var' in self.obj_func:  # ζε€§εη»ι£ι©θ°ζ΄εζΆη
                # ι’ζ΅ζΆηη
                current_holding_pred_return = self.pred_ret_df[date][holding_stock_list].fillna(0).values # TODO: ζΉθΏε€ηnanηζΉζ³
                style_explained_pred_return = X_style_holding_np.T @ np.linalg.inv(X_style_holding_np @ X_style_holding_np.T) @ X_style_holding_np @ current_holding_pred_return
                style_unexplained_pred_return = current_holding_pred_return - style_explained_pred_return
                style_unexplained_pred_return = np.array([style_unexplained_pred_return]).T  # ε’εΉΏ, εζcolumn vector
                q = matrix(- current_holding_pred_return)

                P = matrix(
                    2 * Sigma_holding_stock * self.penalty_lambda \
                  + 2 * style_unexplained_pred_return @ style_unexplained_pred_return.T * self.penalty_xi
                )
            else:
                raise NotImplementedError
                print('Cannot support this objective function')

            # ηΊ¦ζζ‘δ»ΆοΌζδ»ζιδΈδΈι
            G = matrix(np.vstack((np.identity(num_holding), -np.identity(num_holding))))
            h = matrix(np.hstack((np.full(num_holding, weight_high), np.full(num_holding, -weight_low))))
            A = matrix(np.ones(num_holding), (1,num_holding))
            b = matrix(1.0)
            
            # ηΊ¦ζζ‘δ»Ά
            # ι£ζ ΌδΈ­ζ§
            if cfg.style_neutralize:
                benchmark_current_style = np.matmul(X_style_benchmark, benchmark_current_weight).values  # εΊεζζ°ι£ζ Ό
                style_num = X_style.shape[1]
                style_low = np.matrix(np.full(style_num, style_low_limit) + benchmark_current_style).T
                style_high = np.matrix(np.full(style_num, style_high_limit) + benchmark_current_style).T
                G = matrix(np.vstack((G,X_style_holding.values, -X_style_holding.values)))
                h = matrix(np.vstack((h, style_high,-style_low)))

            # θ‘δΈδΈ­ζ§
            if cfg.ind_neutralize:
                benchmark_current_ind = np.matmul(self.ind_df.T[benchmark_stock_list], benchmark_current_weight).values
                ind_num = self.ind_df.shape[1]
                ind_low = np.matrix(np.full(ind_num, ind_low_limit) + benchmark_current_ind).T
                ind_high = np.matrix(np.full(ind_num, ind_high_limit) + benchmark_current_ind).T
                G = matrix(np.vstack((G,X_ind_holding.values, -X_ind_holding.values)))
                h = matrix(np.vstack((h, ind_high,-ind_low)))

            # ιεΆζ’ζη
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

                # !!! ζ ζ³convergeηζΆεθ°ζ΄οΌεθ°ζ΄ι£ζ Όζ΄ι²ηΊ¦ζοΌηΆεθ°ζ΄θ‘δΈηΊ¦ζοΌζεθ°ζ΄ζ’ζηιεΆ

                if min(abs(style_low_limit), abs(style_high_limit)) >=1.0:

                    if min(abs(ind_low_limit), abs(ind_high_limit)) >=1.0:

                        if turnover_limit >= 2.0 :
                            # ζζηΊ¦ζε€±θ΄₯
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
        '''ζδ½³ζδ»ζιθ?‘η?'''
        

        sigma_holding = self.sigma_holding_dict[return_type]

        # opt_signal = pd.DataFrame(0, columns=self.date_list_opt, index=self.tickers)
        self.obj_func = cfg.obj_func
        # weight_low = self.weight_low
        # weight_high = self.weight_high
        
        # θ―»ε₯ι’ζ΅ζΆηη
        self.pred_ret_df = self.read_ml_factor_data(cfg.ml_factor_name, self.tickers, self.date_list)
        # fail_date_list =[]

        # ιζ©δΌεζ¨‘ε
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

        # εΌε§εΎͺη―
        opt_signal_list = []
        for date in self.date_list_opt:
            # extract today data
            Sigma_holding_stock = sigma_holding[date]
            today_holdings = self.holding_stock_list_dict[date]
            prev_holdings_today = prev_holdings[today_holdings].fillna(0)
            # print(date, '\n\n', Sigma_holding_stock, '\n\n', prev_holdings)
            weight = qp_method_func(date, Sigma_holding_stock, prev_holdings_today.values)

            # append to weight
            # opt_signal[date][today_holdings] = weight

            # update prev_holdings
            prev_holdings = pd.Series(weight, index=today_holdings)
            prev_holdings = prev_holdings + template_series
            opt_signal_list.append(prev_holdings.fillna(0).rename(date))

            # print by month
            cur_year_date = date[:6]
            if cur_year_date != prev_year_date:
                # log 
                print(f'{prev_year_date} takes {time.time() - start:.2f} s')
                # renew 
                prev_year_date = cur_year_date
                start = time.time() 
        opt_signal = pd.concat(opt_signal_list, axis=1)
        opt_signal = opt_signal.loc[self.tickers]  # sort 
        self.opt_signal_dict[return_type] = opt_signal



    def save_opt_signal(self):
        """
        ε¨ε­δΌεζι
        """
        output_signal_path = cfg.output_signal_path
        for return_type in self.return_type_list:
            if not os.path.isdir(output_signal_path):
                os.mkdir(output_signal_path)

            file_name = '{}/{}{}{}_{}_{}_{}'.format(
                output_signal_path,
                cfg.input_signal_df_name,  
                cfg.adj_method,
                self.weight_low, 
                self.weight_high, 
                return_type,
                self.obj_func,
            )
            if 'ret_var' in self.obj_func:
                file_name = file_name + f'_penalty{self.penalty_lambda}'
            file_name = file_name + f'_method{self.qp_method}'
            if self.qp_method == 1: 
                file_name = (
                    file_name + 
                    cfg.style_neutralize * f'_style{self.all_style_high_limit}' + 
                    cfg.ind_neutralize * f'_ind{self.all_ind_high_limit}' + 
                    cfg.turnover_constraint * f'_turnover{self.all_turnover_limit}'
                )
            elif self.qp_method == 2:
                file_name = (
                    file_name + 
                    f'_cost_penalty{self.penalty_theta}' + 
                    cfg.style_neutralize * f'_style{self.all_style_high_limit}' + 
                    cfg.ind_neutralize * f'_ind{self.all_ind_high_limit}'
                )
            elif self.qp_method == 3:
                file_name = (
                    file_name + 
                    f'_style_penalty{self.penalty_nu}' + 
                    cfg.ind_neutralize * f'_ind{self.all_ind_high_limit}'
                )
            elif self.qp_method == 4: 
                file_name = (
                    file_name + 
                    f'_misalign_penalty{self.penalty_xi}' + 
                    cfg.style_neutralize * f'_style{self.all_style_high_limit}' + 
                    cfg.ind_neutralize * f'_ind{self.all_ind_high_limit}' + 
                    cfg.turnover_constraint * f'_turnover{self.all_turnover_limit}'
                )
            if self.use_dynamic_ind:
                file_name += self.dynamic_ind_name
            # self.opt_signal_dict[return_type].to_csv(file_name + '.csv')
            # self.myconnector.save_eod_feature(
            #     file_name, self.opt_signal_dict[return_type],
            # )
            self.opt_signal_dict[return_type].reset_index().to_feather(file_name)
            print(file_name)

            # TODO: θΎεΊζδ»Άεε€ͺιΏδΊοΌζ―ε¦θΎεΊηΌε·οΌηΆεθͺε·±θ?°ε½εζ°οΌ


    def start_weight_optimize_process(self):
        print("ζ­£ε¨θ―»εζ°ζ?")
        t0 = time.time()
        self.load_history()
        self.prepare_data()
        print("θ―»εζ°ζ?θζΆ", time.time() - t0)

        print("ζ­£ε¨θ?‘η?ζδ»ζι")
        t0 = time.time()
        for ret_type in self.return_type_list:
            self.cal_sigma_holding(ret_type)
            self.cal_opt_weight(ret_type)

        print("θ?‘η?ζδ»ζιθζΆ", time.time() - t0)
        
        # ζ°ζ?η»ζε¨ε­
        print("ζ­£ε¨ε¨ε­ζ°ζ?")
        t0 = time.time()
        self.save_opt_signal()
        print("ε¨ε­ζ°ζ?θζΆ", time.time() - t0)


