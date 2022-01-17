"""
All Implemented Models for combining factors, including 
- Linear 
- RandomForest
- XGBoost
- LightGBM 
"""

# load packages 
import os
import sys
import time
import copy
import pickle
import numpy as np
import pandas as pd
from itertools import product

# models 
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer

import xgboost as xgb
import lightgbm as lgb

# eval 
from scipy.stats import t, rankdata
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# load files 
from src.factor_combination.configuration import config as cfg
from src.factor_combination.tools.DataTools import DataTools
from src.factor_combination.tools.Evaluation import (
    R2, adjR2, IC, smIC, RMSE, error_rate, slope, IC_cs, smIC_cs, best_group_return
)

# data source 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

'''
————————————————— Linear Model —————————————————
'''

class LinearModel:
    def __init__(self, model_type='linear', alpha=0):
        self.train_mode = cfg.train_mode

        # datatools 
        self.DataTools = DataTools(predict_type=cfg.predict_type)

        # load config 
        self.date_list = self.DataTools.date_list
        self.stock_pool = self.DataTools.stock_pool
        self.features = cfg.features
        self.train_period = cfg.train_period
        self.test_period = cfg.test_period
        self.train_pos = 0
        self.type = cfg.predict_type
        self.model_type = model_type
        self.alpha = alpha
        self.bisection = cfg.bisection
        self.log_path = cfg.log_path
        self.save_log = cfg.save_log
        self.model_path = cfg.model_path
        self.model_params_path = cfg.model_params_path
        self.whether_importance = cfg.whether_importance
        self.weight_method = cfg.weight_method
        self.test_delay = cfg.return_start_date - cfg.return_end_date
        self.model_record_path = cfg.model_record_path

    def myLinearModel(self, paras=None):
        """
        specify model type (linear model type)
        """
        # TODO: read from config 
        # * linear model parameters 
        default_paras = {
            'alpha': 0.001
        }

        if paras is not None:
            for k in paras.keys():
                default_paras[k] = paras[k]

        if 'Regress' in self.type:
            default_paras['eval_metric'] = 'rmse'
            if 'linear' in self.model_type:
                Linear = linear_model.LinearRegression()
            elif 'lasso' in self.model_type:
                Linear = linear_model.Lasso(alpha=self.alpha, max_iter=5000)
            elif 'ridge' in self.model_type:
                Linear = linear_model.Ridge(alpha=self.alpha, max_iter=5000)
            else:
                raise NotImplementedError('Not Such Model Type')

        elif 'Class' in self.type:
            default_paras['eval_metric'] = 'error'
            if 'linear' in self.model_type:
                Linear = linear_model.LogisticRegression()
            elif 'lasso' in self.model_type:
                Linear = linear_model.LogisticRegression(penalty='l1', C=1 / self.alpha, max_iter=5000, solver='saga')
            elif 'ridge' in self.model_type:
                Linear = linear_model.LogisticRegression(penalty='l2', C=1 / self.alpha, max_iter=5000)
        else:
            print("Wrong Type in Linear Training. Use Regressor.")
            default_paras['eval_metric'] = 'rmse'
            if 'linear' in self.model_type:
                Linear = linear_model.LinearRegression()
            elif 'lasso' in self.model_type:
                Linear = linear_model.Lasso(alpha=self.alpha, max_iter=5000)
            elif 'ridge' in self.model_type:
                Linear = linear_model.Ridge(alpha=self.alpha, max_iter=5000)

        return Linear

    def LinearTrain(self):
        """
        Linear Model Main 
        """
        print(f"Linear Training")
        print(f"Train Mode = {self.train_mode}")
        print(f"Train Type = {self.type}")
        print(f"Regularization = {self.model_type}")
        print(f"Date to be trained: {self.date_list[0]} to {self.date_list[-1]}")
        print(f"Using Stock Pool = {cfg.fixed_pool}")
        print(f"Feature Num = {len(self.features)}")
        # get current time 
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M', time.localtime(time.time()))
        self.number = curr_date + curr_time

        self.LinearDef = self.myLinearModel()
        self.number = curr_date + curr_time
        self.model = {}  # 用来存每次滚动训练的线性模型
        self.total_test_y = []  # 用来存每次训练魔性的原始label
        self.total_pred_test_y = []  # 用来存每次训练魔性的预测label

        # 用来储存参数的p值
        self.coef_p_df = pd.DataFrame(
            index=self.features + ['intercept'],
            columns=None, 
            dtype='float'
        )

        self.coef_df = pd.DataFrame(
            index=self.DataTools.features,
            columns=None,
            dtype='float'
        )

        # store labels for the entire test time, index = stock_pool, columns = predict_period
        self.total_test_period_df = pd.DataFrame(
            index=self.stock_pool,
            columns=self.date_list[self.train_period:], 
            dtype='float'
        )

        # self.fig_save_path = cfg.fig_path + 'linear_{}/'.format(self.number)
        # if not os.path.isdir(self.fig_save_path):
        #     os.mkdir(self.fig_save_path)

        while (self.train_pos + self.train_period + self.test_delay + cfg.lag_date) <= len(self.date_list):
            t0 = time.time()

            # specify time frame for each testing mode
            if "roll" in self.train_mode:
                train_date_list = self.date_list[self.train_pos:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "expand" in self.train_mode:
                train_date_list = self.date_list[:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "bisect" in self.train_mode:
                train_date_list = self.date_list[self.train_period:int(len(self.date_list) * self.bisection)]
                test_date_list = train_date_list
                # test_date_list = self.date_list[int(len(self.date_list) * self.bisection):]
            else:
                raise NotImplementedError(f'Not Such Testing Mode {self.train_mode}')
            print(f"\nTraining Linear Model For {train_date_list[0]} to {train_date_list[-1]}")

            # train test split 
            self.DataTools.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
            print('preparation is over.')

            if 'Class' in self.type:
                train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
                test_orig_Y = self.DataTools.test_orig_Y
                train_orig_Y = self.DataTools.train_orig_Y_mask
            else:
                train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
                train_orig_Y = self.DataTools.train_orig_Y_mask

            # for regression, multiply by 100 for numerical precision 
            if 'Regress' in self.type:
                train_Y = train_Y * 100
                test_Y = test_Y * 100
            print(f'Mean Train Y = {np.nanmean(train_Y.values)}')
            print(f'Train X Shape:{train_X.shape}')

            # prepare to be stored 
            pred_test_Y = test_Y.copy(deep=True)

            # obtain weights in dates s=
            weights = self.DataTools.weights_cal(train_Y, self.weight_method)

            # Start Training 
            Linear = self.LinearDef
            Linear.feature_names = self.features
            Linear.fit(train_X.values, train_Y.values, sample_weight=weights)

            # 备用：前向变量逐步回归代码
            # Linear, feature_used = self.linear_forward_select(pd.concat([train_X, train_Y], axis = 1), 'label')

            # 得到参数的p值
            #  if self.save_log:
            #      p_value = self.linear_importance(np.c_[train_X.values, np.ones(train_X.shape[0])], train_Y.values,
            #                                       np.r_[Linear.coef_, Linear.intercept_])
            #      self.coef_p_df['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = p_value

            # obtain coefficients 
            if self.save_log:
                coeff = Linear.coef_
                if 'Regress' in self.type:
                    self.coef_df['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = coeff
                elif 'Class' in self.type:
                    self.coef_df['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = coeff[0]

            # save models 
            self.model['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = copy.deepcopy(Linear)


            # Start Prediction 
            print(f"Testing Linear Model For {test_date_list[0]} to {test_date_list[-1]}")
            self.total_test_y.append(test_Y.values)
            if 'Class' in self.type:
                pred_test_y = Linear.predict_proba(test_X.values)[:, 1]
                pred_test_class_y = Linear.predict(test_X.values)
            else:
                pred_test_y = Linear.predict(test_X.values).reshape(-1)
            pred_test_Y.iloc[:] = pred_test_y
            self.total_test_period_df[test_date_list[cfg.lag_date - 1:]] = pred_test_Y.unstack()
            self.total_pred_test_y.append(pred_test_y)

            # 使用前向选择后的模型进行预测
            # print(f"Testing Linear Model by forward selection For {test_date_list[0]} to {test_date_list[-1]}")
            # self.total_test_y.append(test_Y.values)
            # pred_test_y = Linear.predict(test_X[feature_used]).values.reshape(-1)
            # pred_test_Y.iloc[:] = pred_test_y
            # self.total_test_period_df[test_date_list] = pred_test_Y.unstack()
            # self.total_pred_test_y.append(pred_test_y)

            # compute stats for out-of-sample dataset
            test_Y = test_Y.values.reshape(-1)
            idx = ~(np.isnan(test_Y) | np.isinf(test_Y) | np.isnan(pred_test_y) | np.isinf(pred_test_y))

            # plot in-sample/out-sample stats
            pred_train_y = Linear.predict(train_X)
            train_y = train_Y.values
            # print(type(pred_train_y), type(train_y))
            # self.plot_fig(predy=pred_test_y[idx], testy=test_Y[idx], date=train_date_list[-1], type='outsample')
            # self.plot_fig(predy=pred_train_y, testy=train_y, date=train_date_list[-1], type='insample')

            if 'Regress' in self.type:
                try:
                    print(f'Length = {len(idx)}')
                    r2 = R2(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    ic = IC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    rmse = RMSE(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    print(f'Out Sample R2 = {r2}, IC = {ic}, RMSE = {rmse}')
                except:
                    r2 = 0
                    ic = 0
                    rmse = 0
                    print('no valid test result.')
            elif 'Class' in self.type:
                try:
                    error_arr = error_rate(pred_y=pred_test_class_y[idx], orig_y=test_Y[idx])
                    print(f'Out Sample accuracy = {error_arr[0]}, precision = {error_arr[1]}, '
                          f'recall = {error_arr[2]}, f1_score = {error_arr[3]}')
                except:
                    print('no valid test result.')
            t1 = time.time()
            print(f"Using Time = {t1 - t0}")
            self.train_pos += self.test_period

            if 'bisect' in self.train_mode:
                break

        # backup: empirical adjustments for predicted y values 
        # df['pred_test_y'] = df['pred_test_y'].clip(lower=-30, upper=30)
        # df.to_csv(log_dir + '/y_recorder.csv', index=False)


        with open(os.path.join(self.model_path, 'linear_{}.pkl'.format(self.number)), 'wb') as f:
            pickle.dump(self.model, f)
        
        # save prediction as a factor 
        self.factor_name = "lr_bm_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.date_list[self.train_period][2:],
            self.date_list[-1][2:],
            self.train_period, self.test_period, self.number,
            self.type, self.model_type, self.alpha,
            self.train_mode
        )
        self.DataTools.save_factor_data(
            feature_name=self.factor_name, 
            eod_df=self.total_test_period_df,
            factor_type="ml_factors"
        )
        print(self.factor_name)

        # save coefficients 
        if self.save_log:
            with open(os.path.join(
                self.log_path,'linear_coef_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(
                    self.date_list[self.train_period][2:], self.date_list[-1][2:],
                    self.train_period, self.test_period, self.number, self.type, self.model_type, self.alpha,
                    self.train_mode)), 'wb') as f:
                pickle.dump(self.coef_df, f)

        return self.total_test_y, self.total_pred_test_y

    # def linear_importance(self, X, y, beta):
    #     eps = (y - np.matmul(X, beta))
    #     # eps = eps.reshape(len(eps), 1)
    #     sigma_hat = np.sum(eps ** 2) / (X.shape[0] - X.shape[1])
    #     var_beta = sigma_hat * np.linalg.inv(np.matmul(X.T, X))
    #     var_beta = np.diagonal(var_beta)
    #     # 得到t值
    #     beta_t = beta / np.sqrt(var_beta)
    #     # 得到p值
    #     beta_p = t.sf(beta_t, X.shape[0] - X.shape[1])
    #     beta_p = 2 * (0.5 - np.abs(beta_p - 0.5))
    #     return beta_t

    # TODO: documentation 
    def plot_fig(self, predy, testy, date, type):
        plt.figure(figsize=(10, 10), dpi=200)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        draw_x = testy
        draw_y = predy - testy
        p1 = plt.scatter(draw_x[np.where(draw_x > 0)],
                         draw_y[np.where(draw_x > 0)], s=2, color='lightcoral', alpha=0.3)
        p2 = plt.scatter(draw_x[np.where(draw_x <= 0)],
                         draw_y[np.where(draw_x <= 0)], s=2, color='royalblue', alpha=0.3)
        axis_x = np.linspace(-5, 5, 10)
        zero_line = np.linspace(0, 0, 10)
        plt.plot(axis_x, zero_line, linestyle='--', color='darkgrey')
        plt.legend(handles=[p1, p2], labels=['pos_stocks', 'neg_stocks'], loc='best',
                   fontsize=12)
        plt.savefig(self.fig_save_path + 'linear_scatter_{}_{}_{}.png'.format(self.number, date, type))
        plt.close()

    # # 定义向前逐步回归函数
    # def linear_forward_select(self, data, target):
    #     variate = set(data.columns)  # 将字段名转换成字典类型
    #     variate.remove(target)  # 去掉因变量的字段名
    #     selected = []
    #     current_score, best_new_score = float('inf'), float('inf')  # 目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
    #     # 循环筛选变量
    #     while variate:
    #         aic_with_variate = []
    #         for candidate in variate:  # 逐个遍历自变量
    #             formula = "{}~{}".format(target, "+".join(selected + [candidate]))  # 将自变量名连接起来
    #             aic = ols(formula=formula, data=data).fit().aic  # 利用ols训练模型得出aic值
    #             aic_with_variate.append((aic, candidate))  # 将第每一次的aic值放进空列表
    #         aic_with_variate.sort(reverse=True)  # 降序排序aic值
    #         # 最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
    #         best_new_score, best_candidate = aic_with_variate.pop()
    #         if current_score > best_new_score:  # 如果目前的aic值大于最好的aic值
    #             variate.remove(best_candidate)  # 移除加进来的变量名，即第二次循环时，不考虑此自变量了
    #             selected.append(best_candidate)  # 将此自变量作为加进模型中的自变量
    #             current_score = best_new_score  # 最新的分数等于最好的分数
    #             print("aic is {},continuing!".format(current_score))  # 输出最小的aic值
    #         else:
    #             print("This model's selection over!")
    #             break
    #     formula = "{}~{}".format(target, " + ".join(selected))  # 最终的模型式子
    #     print("final formula is {}".format(formula))
    #     model = ols(formula=formula, data=data).fit()
    #     return model, selected

    # def Linear_Para_Search(self):
    #     """
    #     parameter search for linear models 
    #     """
    #     print(f"Linear Parameter Search by Cross Validation")
    #     print(f"Train Type = {self.type}")
    #     print(f"Regularization = {self.model_type}")
    #     print(f"Metric = {cfg.score_metric}")
    #     print(f"Date to be trained: {self.date_list[0]} to {self.date_list[-1]}")
    #     print(f"Using Stock Pool = {cfg.index}")
    #     print(f"Feature Num = {len(self.features)}")

    #     train_date_list = self.date_list
    #     test_date_list = train_date_list

    #     # 获取现在的时间，用来记录模型编号
    #     curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
    #     curr_time = time.strftime('%H%M', time.localtime(time.time()))
    #     self.number = curr_date + curr_time

    #     self.DataTools.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
    #     print('preparation is over.')
    #     if 'Class' in self.type:
    #         train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
    #         test_orig_Y = self.DataTools.test_orig_Y
    #         train_orig_Y = self.DataTools.train_orig_Y_mask
    #     else:
    #         train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
    #         train_orig_Y = self.DataTools.train_orig_Y_mask

    #     # 对label的值乘以100，避免数值精度影响模型结果
    #     if 'Regress' in self.type:
    #         train_Y = train_Y * 100
    #         test_Y = test_Y * 100
    #     print(f'Mean Train Y = {np.nanmean(train_Y.values)}')
    #     print(f'Train X Shape:{train_X.shape}')

    #     # 确定训练的sample_weights, weight_method在config中设置，'EW'即default等权
    #     weights = self.DataTools.weights_cal(train_Y, self.weight_method)

    #     if 'Regress' in self.type:
    #         if 'lasso' in self.model_type:
    #             Linear_Para = linear_model.Lasso(max_iter=5000, random_state=21)
    #         elif 'ridge' in self.model_type:
    #             Linear_Para = linear_model.Ridge(max_iter=5000, random_state=21)
    #         # else:
    #         #    print('...Error:{}'.format(self.model_type))
    #         #    break

    #     elif 'Class' in self.type:
    #         if 'lasso' in self.model_type:
    #             Linear_Para = linear_model.LogisticRegression(penalty='l1', max_iter=5000, solver='saga',
    #                                                           random_state=21)
    #         elif 'ridge' in self.model_type:
    #             Linear_Para = linear_model.LogisticRegression(penalty='l2', max_iter=5000, random_state=21)

    #     alpha_range = cfg.alpha_range

    #     if cfg.self_defined_score == True:
    #         score_function = make_scorer(eval(cfg.score_metric))
    #     else:
    #         score_function = cfg.score_metric

    #     t0 = time.time()

    #     if 'Regress' in self.type:
    #         grid = GridSearchCV(Linear_Para, param_grid={'alpha': alpha_range}, scoring=score_function, cv=2,
    #                             refit=False).fit(train_X, train_Y, sample_weight=weights)
    #         print(f"Best parameters = {grid.best_params_}")
    #     elif 'Class' in self.type:
    #         if 'best_group_return' in cfg.score_metric:
    #             kf = KFold(n_splits=2)
    #             CV_record = []
    #             for p in alpha_range:
    #                 ret = 0
    #                 # For each set of params, fit on CV set
    #                 for train, test in kf.split(train_orig_Y):
    #                     Train_X_k = train_X.iloc[train,]
    #                     Train_Y_k = train_Y[train]
    #                     # Train_orig_Y_k = Train_orig_Y[train]
    #                     Test_X_k = train_X.iloc[test,]
    #                     # Test_Y_k = Train_Y[test]
    #                     Test_orig_Y_k = train_orig_Y[test]
    #                     if 'lasso' in self.model_type:
    #                         Linear_Para = linear_model.LogisticRegression(C=1 / p, penalty='l1', max_iter=5000,
    #                                                                       solver='saga', random_state=21)
    #                     elif 'ridge' in self.model_type:
    #                         Linear_Para = linear_model.LogisticRegression(C=1 / p, penalty='l2', max_iter=5000,
    #                                                                       random_state=21)
    #                     Linear_Para.fit(Train_X_k, Train_Y_k)
    #                     pred_test_y = Linear_Para.predict_proba(Test_X_k)[:, 1]
    #                     r = best_group_return(Test_orig_Y_k, pred_test_y)
    #                     ret = ret + r
    #                 CV_record.append(ret)
    #             print(f"Best parameters = {alpha_range[np.argmax(CV_record)]}")

    #         else:
    #             grid = GridSearchCV(Linear_Para, param_grid={'C': 1 / alpha_range}, scoring=score_function, cv=2,
    #                                 refit=False).fit(train_X, train_Y, sample_weight=weights)
    #             print(f"Best parameter alpha = {1 / grid.best_params_['C']}")

    #     # 得到CV结果
    #     # print(f"CV result = {grid.cv_results_['mean_test_score']}")
    #     if self.save_log:
    #         if ('Class' in self.type) & ('best_group_return' in cfg.score_metric):
    #             self.cv_result_df = pd.DataFrame({'params': alpha_range, 'Test Score': CV_record})
    #         else:
    #             self.cv_result_df = pd.DataFrame(
    #                 {'params': alpha_range, 'Test Score': grid.cv_results_['mean_test_score']})
    #         with open(self.log_path + 'linear_CV_{}_{}_{}_{}.pkl'.format(self.date_list[0][2:], self.date_list[-1][2:],
    #                                                                      self.number, self.type), 'wb') as f:
    #             pickle.dump(self.cv_result_df, f)

    #     #  if 'Regress' in self.type:
    #     #      self.coef_df['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = coeff
    #     #  elif 'Class' in self.type:
    #     #      self.coef_df['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = coeff[0]

    #     t1 = time.time()
    #     print(f"Using Time = {t1 - t0}")

    #     return self.cv_result_df

'''
————————————————— XGBoost —————————————————
'''

class XgbModel:
    def __init__(self):
        self.train_mode = cfg.train_mode
        # datatools 
        self.DataTools = DataTools(predict_type=cfg.predict_type)

        # load configs
        self.date_list = self.DataTools.date_list
        self.stock_pool = self.DataTools.stock_pool
        self.features = cfg.features
        # self.gene_features = self.DataTools.gene_features
        # self.feature_dict = cfg.feature_dict
        self.train_period = cfg.train_period
        self.test_period = cfg.test_period
        self.train_pos = 0
        self.xgb_paras = cfg.xgb_paras
        self.type = cfg.predict_type
        self.nest_type = cfg.nest_type
        self.bisection = cfg.bisection
        self.log_path = cfg.log_path
        self.model_path = cfg.model_path
        self.model_params_path = cfg.model_params_path
        self.whether_importance = cfg.whether_importance
        self.weight_method = cfg.weight_method
        self.test_delay = cfg.return_start_date - cfg.return_end_date
        self.model_record_path = cfg.model_record_path

    def myXgbModel(self, paras=None, type='Regressor'):
        """ define XGBoost parameters """
        default_paras = paras
        default_paras['objective'] = 'reg:squarederror'
        XgbParams = {
            'num_boost_round': default_paras['num_round'],
            'verbose_eval': int(default_paras['num_round'] / 5),
            'maximize': False,
        }
        default_paras.pop('num_round')
        if 'Regress' in type:
            XgbParams['params'] = default_paras
            # XgbParams['obj'] = self.xgbreg_loss_function
            XgbParams['feval'] = self.xgb_feval_func
        elif 'Class' in type:
            default_paras['objective'] = 'binary:hinge'
            XgbParams['params'] = default_paras
        else:
            print("Wrong Type in XgBoost Training. Use Regressor instead.")
            XgbParams['params'] = default_paras
            XgbParams['obj'] = self.xgbreg_loss_function
            XgbParams['feval'] = self.xgb_feval_func

        return XgbParams

    def NestedXgbTrain(self):
        raise NotImplementedError('Nested to be fixed')

        print(f"Nested XGB Training")
        print(f"Train Mode = {self.train_mode}")
        print(f"Date to be trained: {self.date_list[0]} to {self.date_list[-1]}")
        print(f"Using Stock Pool = {cfg.fixed_pool}")
        print(f"Whether Neutralize = {str(cfg.neutralize)}")
        print(f"Feature Num Part 1 = {len(self.feature_dict[0])}")
        print(f"Feature Num Part 2 = {len(self.feature_dict[1])}")
        # 获取现在的时间，用来记录模型编号
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M', time.localtime(time.time()))

        self.DataTools_1 = DataTools(offline=self.offline, predict_type=self.nest_type[0], features=cfg.feature_dict[0])
        self.DataTools_2 = DataTools(offline=self.offline, predict_type=self.nest_type[1], features=cfg.feature_dict[1])
        self.XgbPara1 = self.myXgbModel(paras=cfg.nest_xgb_paras_lst[0], type=self.nest_type[0])
        self.XgbPara2 = self.myXgbModel(paras=cfg.nest_xgb_paras_lst[1], type=self.nest_type[1])
        self.number = curr_date + curr_time
        print(f"Model Number = {self.number}")
        self.model = {}  # 用来存每次滚动训练的线性模型
        self.total_test_y = []  # 用来存每次训练魔性的原始label
        self.total_pred_test_y = []  # 用来存每次训练魔性的预测label
        self.importance_dict = {}  # 用来存每次模型特征重要性

        self.result_dict = {} # 结果储存字典
        self.result_dict['in_IC_list'] = [] # 用来存每次模型的样本内IC
        self.result_dict['in_rankIC_list'] = [] # 用来存每次模型的样本内rankIC
        self.result_dict['in_rmse_list'] = [] # 用来存每次模型的样本内rmse
        self.result_dict['out_IC_list'] = [] # 用来存每次模型的样本外IC
        self.result_dict['out_rankIC_list'] = [] # 用来存每次模型的样本外rankIC
        self.result_dict['out_rmse_list'] = [] # 用来存每次模型的样本外rmse
        self.result_dict['period_list'] = [] # 用来做画图的横坐标

        # 用来存整个预测区间的预测label(作为因子值)的dataframe, index = stock_pool, columns = predict_period
        self.total_test_period_df = pd.DataFrame(index=self.stock_pool,
                                                 columns=self.date_list[self.train_period:], dtype='float')

        self.total_test_mask_df = pd.DataFrame(index=self.stock_pool,
                                                 columns=self.date_list[self.train_period:], dtype='float')

        self.fig_save_path = cfg.fig_path + 'xgb_fig_{}/'.format(self.number)
        if not os.path.isdir(self.fig_save_path):
            os.mkdir(self.fig_save_path)

        # 记录模型训练的各种信息
        others = 'vague_zero = 0.01, vague_large = 1.5, '  # 备注
        record_info = [cfg.start_date, cfg.end_date, str(len(self.features)), cfg.fixed_pool, self.train_mode,
                       self.train_period, self.test_period, self.bisection, cfg.return_type, cfg.return_start_date,
                       cfg.return_end_date, cfg.nest_weight_method, cfg.nest_type, cfg.lag_date, int(cfg.valid_y), others]
        if os.path.exists(self.model_record_path):
            record_df = pd.read_csv(self.model_record_path, index_col=0)
            record_df.loc[self.number] = record_info
            # record_df.to_csv(self.model_record_path)
        else:
            new_record_df = pd.DataFrame(columns=['startdate', 'enddate', 'features', 'stockpool',
                                                  'trainmode', 'trainperiod', 'testperiod', 'bisection', 'returntype',
                                                  'sampleweight', 'others'])
            new_record_df.loc[self.number] = record_info
            # new_record_df.to_csv(self.model_record_path)


        t_start = time.time()

        # 总训练开始
        while (self.train_pos + self.train_period + self.test_delay + cfg.lag_date) <= len(self.date_list):
            t01 = time.time()

            # 按照训练模式设定每次滚动的训练时间区间和预测时间区间
            if "roll" in self.train_mode:
                train_date_list = self.date_list[self.train_pos:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "expand" in self.train_mode:
                train_date_list = self.date_list[:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "bisect" in self.train_mode:
                train_date_list = self.date_list[:int(len(self.date_list) * self.bisection) + cfg.lag_date - 1]
                test_date_list = self.date_list[int(len(self.date_list) * self.bisection) + self.test_delay:]
            else:
                print("...Error:{}...".format(self.train_mode))
                break
            print(f"\n***********Training Xgb Model For {train_date_list[0]} to {train_date_list[-1]}***********")

            # 准备第一层模型的数据
            self.result_dict['period_list'].append(test_date_list[-1])
            self.DataTools_1.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
            t2 = time.time()
            print(f"> Time: Load Outer Model Data = {round(t2 - t01)}s")
            train_X, train_Y, test_X, test_Y = self.DataTools_1.standard_clean_data()
            test_orig_Y = self.DataTools_1.test_orig_Y
            train_orig_Y = self.DataTools_1.train_orig_Y
            train_pred_Y = train_orig_Y.copy(deep=True)
            # 准备非nan的mask
            na_idx = self.DataTools_1.na_idx
            if cfg.diy_return:
                self.y_mask = self.DataTools_1.test_mask

            t3 = time.time()
            print('Preparation for outer model is over.')
            print(f"> Time: Prepare Outer Model Data = {round(t3 - t2)}s\n")

            # 对label的值乘以10，避免数值精度影响模型结果
            if 'Regress' in self.nest_type[0]:
                train_Y = train_Y * 10
                test_Y = test_Y * 10
            print(f'Outer Model Mean Train Y = {np.nanmean(train_Y.values)}')
            print(f'Outer Model Train X Shape:{train_X.shape}')

            # 准备存储预测结果的dataframe
            pred_test_Y = test_Y.copy(deep=True)

            # 确定训练的sample_weights, weight_method在config中设置，'EW'即default等权
            weights_1 = self.DataTools_1.weights_cal(Y=train_Y, method=cfg.nest_weight_method[0], na_idx=na_idx)

            # 训练XgBoost模型
            train_xgb_1 = xgb.DMatrix(data=train_X, label=train_Y, weight=weights_1)
            test_xgb_1 = xgb.DMatrix(data=test_X, label=test_Y)
            watch_list_1 = [(train_xgb_1, 'train'), (test_xgb_1, 'test')]
            XgbTree_1 = xgb.train(**self.XgbPara1, dtrain=train_xgb_1, evals=watch_list_1)

            t4 = time.time()
            print(f"> Time: Train Outer Model = {round(t4 - t3)}s\n")

            # 存储模型
            self.model['train{}_{}_outer'.format(train_date_list[0], train_date_list[-1])] = copy.deepcopy(XgbTree_1)
            with open(self.model_path + 'xgb_model_outer.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            # 存储特征重要性log
            if self.whether_importance:
                self.importance_df = self.xgb_importance(XgbTree_1)
                self.importance_dict['train{}_{}_outer'.format(train_date_list[0], train_date_list[-1])] = self.importance_df

            t_gen_start = time.time()
            # 使用XgBoost模型进行预测
            print(f"Testing Outer Xgb Model For {test_date_list[cfg.lag_date - 1]} to {test_date_list[-1]}.")
            print(f"Generating Rubish Stock Mask.")
            self.total_test_y.append(test_Y)
            if 'Class' in self.nest_type[0]:
                pred_test_y = np.array(XgbTree_1.predict_proba(test_xgb_1))[:, 1]
                pred_test_class_y = np.array(XgbTree_1.predict(test_xgb_1))
            else:
                pred_test_y = np.array(XgbTree_1.predict(test_xgb_1))
            pred_test_Y.iloc[:] = pred_test_y
            # 第一层预测的结果将test_period中每天绩效差的票加一层mask，用WeedOutMask函数剔除，阈值在函数中设置
            self.total_test_mask_df[test_date_list[cfg.lag_date - 1:]] = self.WeedOutMask(pred_test_Y.unstack())
            t_gen_end = time.time()
            print(f"> Time: Generate Mask = {round(t_gen_end - t_gen_start)}s\n")

            # 准备第二层模型的训练数据
            t02 = time.time()
            '''
            self.DataTools_2.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
            t2 = time.time()
            print(f"> Time: Load Core Model Data = {round(t2 - t02)}s")
            train_X, train_Y, test_X, test_Y = self.DataTools_2.standard_clean_data()
            test_orig_Y = self.DataTools_2.test_orig_Y
            train_orig_Y = self.DataTools_2.train_orig_Y
            train_pred_Y = train_orig_Y.copy(deep=True)
            # 准备非nan的mask
            na_idx = self.DataTools_2.na_idx
            if not cfg.dynamic_pool:
                self.y_mask = self.DataTools_2.test_mask

            t3 = time.time()
            print('Preparation for core model is over.')
            print(f"> Time: Prepare Core Model Data = {round(t3 - t2)}s\n")

            # 对label的值乘以100，避免数值精度影响模型结果
            if 'Regress' in self.nest_type[1]:
                train_Y = train_Y * 10
                test_Y = test_Y * 10
            print(f'Core Model Mean Train Y = {np.nanmean(train_Y.values)}')
            print(f'Core Model Train X Shape:{train_X.shape}')
            '''
            # 准备存储预测结果的dataframe
            pred_test_Y = test_Y.copy(deep=True)

            # 确定训练的sample_weights, weight_method在config中设置，'EW'即default等权
            self.DataTools_2.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
            weights_2 = self.DataTools_2.weights_cal(Y=train_Y, method=cfg.nest_weight_method[1], na_idx=na_idx)

            # 训练XgBoost模型
            train_xgb_2 = xgb.DMatrix(data=train_X, label=train_Y, weight=weights_2)
            test_xgb_2 = xgb.DMatrix(data=test_X, label=test_Y)
            watch_list_2 = [(train_xgb_2, 'train'), (test_xgb_2, 'test')]
            XgbTree_2 = xgb.train(**self.XgbPara2, dtrain=train_xgb_2, evals=watch_list_2)
            t4 = time.time()
            print("Train Core Model over.")
            print(f"> Time: Train Core Model = {round(t4 - t02)}s\n")

            # 存储模型
            self.model['train{}_{}_core'.format(train_date_list[0], train_date_list[-1])] = copy.deepcopy(XgbTree_2)
            with open(self.model_path + 'xgb_model_core.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            # 存储特征重要性log
            if self.whether_importance:
                self.importance_df = self.xgb_importance(XgbTree_2)
                self.importance_dict['train{}_{}_core'.format(train_date_list[0], train_date_list[-1])] = self.importance_df

            # 使用XgBoost模型进行预测
            print(f"Testing Xgb Model For {test_date_list[cfg.lag_date - 1]} to {test_date_list[-1]}")
            self.total_test_y.append(test_Y)
            self.mask_values = self.total_test_mask_df[test_date_list[cfg.lag_date - 1:]].stack(dropna=False).values
            if 'Class' in self.type:
                pred_test_y = np.array(XgbTree_2.predict_proba(test_X))[:, 1] * self.mask_values # 储存概率作为因子
                pred_test_class_y = np.array(XgbTree_2.predict(test_X)) * self.mask_values # 储存分类结果用于评价模型
                # 是否用动态票池加一层mask
                if cfg.diy_return:
                    pred_test_class_y = pred_test_class_y * self.y_mask
                    pred_test_y = pred_test_y * self.y_mask
            else:
                pred_test_y = np.array(XgbTree_2.predict(test_X)) * self.mask_values
                # 是否用动态票池加一层mask
                if cfg.diy_return:
                    pred_test_y = pred_test_y * self.y_mask
            pred_test_Y.iloc[:] = pred_test_y
            self.total_test_period_df[test_date_list[cfg.lag_date - 1:]] = pred_test_Y.unstack()
            self.total_pred_test_y.append(pred_test_y)

            # 计算样本外的指标
            test_y = test_Y.values.reshape(-1)
            idx = ~(np.isnan(test_y) | np.isinf(test_y) | np.isnan(pred_test_y) | np.isinf(pred_test_y))

            # 临时绘图
            pred_train_y = np.array(XgbTree_2.predict(train_X))
            train_pred_Y.iloc[na_idx[:, 0]] = pred_train_y
            train_y = train_Y.values
            print("====Explanatory Power====")
            self.plot_eps_fig(predy=pred_test_y[idx], testy=test_y[idx], date=train_date_list[-1], type='outsample')
            self.plot_eps_fig(predy=pred_train_y, testy=train_y, date=train_date_list[-1], type='insample')
            self.plot_rank_fig(predy=pred_test_Y.unstack(), testy=test_Y.unstack(), date=train_date_list[-1],
                               type='outsample')

            print("=====Evaluation Stat=====")
            if 'Regress' in self.type:
                try:
                    r2 = R2(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                    # ic = IC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    # smic = smIC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    ic = IC_cs(pred_y=pred_test_Y.unstack(), orig_y=test_orig_Y.unstack())
                    smic = smIC_cs(pred_y=pred_test_Y.unstack(), orig_y=test_orig_Y.unstack())
                    rmse = RMSE(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                    self.result_dict['out_IC_list'].append(ic)
                    self.result_dict['out_rankIC_list'].append(smic)
                    self.result_dict['out_rmse_list'].append(rmse)
                    print(
                        'OutSample R2 = {:.5f}, IC = {:.5f}, rankIC = {:.5f}, RMSE = {:.5f}'.format(r2, ic, smic, rmse))
                    r2 = R2(pred_y=pred_train_y, orig_y=train_y)
                    # ic = IC(pred_y=pred_train_y, orig_y=train_y)
                    # smic = smIC(pred_y=pred_train_y, orig_y=train_y)
                    ic = IC_cs(pred_y=train_pred_Y.unstack(), orig_y=train_orig_Y.unstack())
                    smic = smIC_cs(pred_y=train_pred_Y.unstack(), orig_y=train_orig_Y.unstack())
                    rmse = RMSE(pred_y=pred_train_y, orig_y=train_y)
                    self.result_dict['in_IC_list'].append(ic)
                    self.result_dict['in_rankIC_list'].append(smic)
                    self.result_dict['in_rmse_list'].append(rmse)
                    print(
                        'InSample R2 = {:.5f}, IC = {:.5f}, rankIC = {:.5f}, RMSE = {:.5f}'.format(r2, ic, smic, rmse))
                except:
                    print('no valid test result.')
            elif 'Class' in self.type:
                try:
                    error_arr = error_rate(pred_y=pred_test_class_y[idx], orig_y=test_y[idx])
                    ic = IC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    smic = smIC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    self.result_dict['out_IC_list'].append(ic)
                    self.result_dict['out_rankIC_list'].append(smic)
                    print('Out Sample accuracy = {:.5f}, IC = {:.5f}, rankIC = {:.5f}'.format(error_arr[0], ic, smic))
                    print('Out Sample precision = {:.5f}, recall = {:.5f}, f1_score = {:.5f}'.format(error_arr[1],
                                                                                                     error_arr[2],
                                                                                                     error_arr[3]))

                except:
                    print('no valid test result.')

            t1 = time.time()
            print(f"************************Used Time = {round(t1 - t01)}s************************")
            self.train_pos += self.test_period

            if 'bisect' in self.train_mode:
                break

        # 备用代码：将数据去除极值
        # df['pred_test_y'] = df['pred_test_y'].clip(lower=-30, upper=30)
        # df.to_csv(log_dir + '/y_recorder.csv', index=False)

        t_end = time.time()
        print(f"Total IC = {np.nanmean(np.array(self.result_dict['out_IC_list']))}")
        print(f"Total rank IC = {np.nanmean(np.array(self.result_dict['out_rankIC_list']))}")
        print(f"Time: Total Used Time = {t_end - t_start}")

        # 将预测结果输出成因子 h5 格式
        self.factor_name = "xgb_{}_{}_{}_{}_{}_{}".format(self.date_list[self.train_period][2:], self.date_list[-1][2:],
                                                          self.train_period, self.test_period, self.number, self.type)
        self.DataTools.save_factor_data(feature_name=self.factor_name, eod_df=self.total_test_period_df,
                                        factor_type="ml_factors")

        # 绘制IC/rankIC图
        self.plot_result_fig()

        # 将模型重要性储存
        print(f'Trained Factor Name is {self.factor_name}')
        self.importance_dict['factor_list'] = self.features
        with open(self.log_path + 'xgb_importance_{}.pkl'.format(self.number), 'wb') as f:
            pickle.dump(self.importance_dict, f)

        return self.total_test_y, self.total_pred_test_y

    # # 用来做垃圾股剔除的小函数
    # def WeedOutMask(self, df):
    #     return ((df > df.quantile(0.1)) + (df - df)).replace(0, np.nan)

    def XgbTrain(self):
        """ main xgb """
        print(f"XGB Training")
        print(f"Train Mode = {self.train_mode}")
        print(f"Train Type = {self.type}")
        print(f"Date to be trained: {self.date_list[0]} to {self.date_list[-1]}")
        print(f"Using Stock Pool = {cfg.fixed_pool}")
        print(f"Whether Neutralize = {str(cfg.neutralize)}")
        print(f"Whether DIY Return = {str(cfg.diy_return)}")
        print(f"Sample Weight Method = {cfg.weight_method}")
        # print(f"Feature Num = {len(self.features) + len(self.gene_features)}")
        print(f"Lag Times = {cfg.lag_date} (1 means no lag)")
        # get current time for logging 
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M', time.localtime(time.time()))

        # ? probablly don't need to re-init since already done so above 
        # self.DataTools = DataTools(offline=self.offline, predict_type=cfg.predict_type, features=cfg.features)

        print(f"Feature Num = {len(self.DataTools.features)}")
        self.XgbPara = self.myXgbModel(paras=cfg.xgb_paras, type=self.type)
        self.number = curr_date + curr_time
        print(f"Model Number = {self.number}")
        # with open(self.model_params_path + "xgb_{}.pkl".format(self.number), 'wb') as f:
        #     pickle.dump(self.XgbDef.get_params(), f)
        self.model = {} # 用来存每次滚动训练的线性模型
        self.total_test_y = [] # 用来存每次训练魔性的原始label
        self.total_pred_test_y = [] # 用来存每次训练魔性的预测label
        self.importance_dict = {} # 用来存每次模型特征重要性

        # TODO: what is temp for ?
        # !!! TMP !!!
        self.tmp_train_Y_dict = {}

        self.result_dict = {} # 结果储存字典
        self.result_dict['in_IC_list'] = [] # 用来存每次模型的样本内IC
        self.result_dict['in_rankIC_list'] = [] # 用来存每次模型的样本内rankIC
        self.result_dict['in_rmse_list'] = [] # 用来存每次模型的样本内rmse
        self.result_dict['out_IC_list'] = [] # 用来存每次模型的样本外IC
        self.result_dict['out_rankIC_list'] = [] # 用来存每次模型的样本外rankIC
        self.result_dict['out_rmse_list'] = [] # 用来存每次模型的样本外rmse
        self.result_dict['period_list'] = [] # 用来做画图的横坐标

        # store label in the time frame as a dataframe, index = stock_pool, columns = predict_period
        self.total_test_period_df = pd.DataFrame(
            index = self.stock_pool,
            columns = self.date_list[self.train_period:], 
            dtype = 'float'
        )

        self.fig_save_path = cfg.fig_path + 'xgb_fig_{}/'.format(self.number)
        if not os.path.isdir(self.fig_save_path):
            os.mkdir(self.fig_save_path)

        # # log 
        # others = ''  # backup 
        # record_info = [
        #     cfg.start_date, cfg.end_date, str(len(self.features)), cfg.fixed_pool, self.train_mode,
        #     self.train_period, self.test_period, self.bisection, cfg.return_type, cfg.return_start_date,
        #     cfg.return_end_date, self.weight_method, self.type, cfg.lag_date, int(cfg.valid_y), others
        # ]
        # if os.path.exists(self.model_record_path):
        #     record_df = pd.read_csv(self.model_record_path, index_col=0)
        #     record_df.loc[self.number] = record_info
        #     record_df.to_csv(self.model_record_path)
        # else:
        #     new_record_df = pd.DataFrame(columns = ['startdate', 'enddate', 'features', 'stockpool',
    	#                 'trainmode', 'trainperiod', 'testperiod', 'bisection', 'returntype',
	    #                 'sampleweight', 'others'])
        #     new_record_df.loc[self.number] = record_info
        #     new_record_df.to_csv(self.model_record_path)

        t_start = time.time()

        while (self.train_pos + self.train_period + self.test_delay + cfg.lag_date) <= len(self.date_list):
            t0 = time.time()

            # specify train test time frame 
            if "roll" in self.train_mode:
                train_date_list = self.date_list[self.train_pos:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "expand" in self.train_mode:
                train_date_list = self.date_list[:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "bisect" in self.train_mode:
                train_date_list = self.date_list[:max(int(len(self.date_list) * self.bisection) + cfg.lag_date - 1, len(self.date_list))]
                test_date_list = train_date_list
                # test_date_list = self.date_list[int(len(self.date_list) * self.bisection) + self.test_delay:]
            else:
                # print("...Error:{}...".format(self.train_mode))
                raise NotImplementedError(f'Train mode {self.train_mode} DNE')
            
            # Train 
            print(f"\n********************** Training Xgb Model For {train_date_list[0]} to {train_date_list[-1]} **********************")
            self.xgb_main_train(train_date_list=train_date_list, test_date_list=test_date_list, t0=t0)
            t1 = time.time()
            print(f"************************************ Used Time = {round(t1 - t0)}s ************************************")
            self.train_pos += self.test_period

            if 'bisect' in self.train_mode:
                break

        # backup: empricial adjustments for y values 
        # df['pred_test_y'] = df['pred_test_y'].clip(lower=-30, upper=30)
        # df.to_csv(log_dir + '/y_recorder.csv', index=False)

        # save feather 
        self.factor_name = "xgb_{}_{}_{}_{}_{}_{}".format(
            self.date_list[self.train_period][2:], 
            self.date_list[-1][2:],
            self.train_period, 
            self.test_period, 
            self.number, 
            self.type
        )
        self.DataTools.save_factor_data(
            feature_name=self.factor_name, 
            eod_df=self.total_test_period_df,
            factor_type="ml_factors"
        )

        # plot IC/rank IC 
        self.plot_result_fig()

        # save trained model
        with open(os.path.join(self.model_path, 'xgb_model_{}.pkl'.format(self.number)), 'wb') as f:
            pickle.dump(self.model, f)


        t_end = time.time()
        print(f"Total IC = {np.nanmean(np.array(self.result_dict['out_IC_list']))}")
        print(f"Total rank IC = {np.nanmean(np.array(self.result_dict['out_rankIC_list']))}")
        print(f"Time: Total Used Time = {t_end - t_start}")

        # save feature importnce
        print(f'Trained Factor Name is {self.factor_name}')
        self.importance_dict['factor_list'] = self.features
        self.importance_dict['xgb_paras'] = cfg.xgb_paras
        with open(os.path.join(self.log_path, 'xgb_importance_{}.pkl'.format(self.number)), 'wb') as f:
            pickle.dump(self.importance_dict, f)

        return self.total_test_y, self.total_pred_test_y


    def xgb_main_train(self, train_date_list, test_date_list, t0):
        # prepare data 
        self.DataTools.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
        self.result_dict['period_list'].append(test_date_list[-1])
        t2 = time.time()
        print(f"> Time: Load Data = {round(t2 - t0)}s")

        # if test_date_list[-1] <= '20180303':
        if True:
            # 准备数据
            train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()

            # ################################################ !!!! ############################################################
            # test_Y.fillna(0)
            # ################################################ !!!! ############################################################

            test_orig_Y = self.DataTools.test_orig_Y
            train_orig_Y = self.DataTools.train_orig_Y
            train_pred_Y = train_orig_Y.copy(deep=True)
            self.tmp_train_Y_dict['trainY_{}_{}'.format(train_date_list[0], train_date_list[-1])] = train_Y
            self.tmp_train_Y_dict['origY_{}_{}'.format(train_date_list[0], train_date_list[-1])] = train_orig_Y
            # prepare non-nan mask
            na_idx = self.DataTools.na_idx
            if cfg.diy_return:
                self.y_mask = self.DataTools.test_mask

            t3 = time.time()
            print('preparation is over.')
            print(f"> Time: Prepare Data = {round(t3 - t2)}s\n")

            # multiply 10 for numerical precision
            if 'Regress' in self.type:
                train_Y = train_Y * 10
                test_Y = test_Y * 10
            print(f'Mean Train Y = {np.nanmean(train_Y.values)}')
            print(f'Train X Shape:{train_X.shape}\n')

            # prepare to be saved 
            pred_test_Y = test_Y.copy(deep=True)

            # get weight
            weights = self.DataTools.weights_cal(Y=train_Y, method=self.weight_method, na_idx=na_idx)

            # train 
            print('===================================== Train Log =====================================')
            train_xgb = xgb.DMatrix(data=train_X, label=train_Y, weight=weights)
            test_xgb = xgb.DMatrix(data=test_X, label=test_Y)
            watch_list = [(train_xgb, 'train'), (test_xgb, 'test')]
            XgbTree = xgb.train(**self.XgbPara, dtrain=train_xgb, evals=watch_list)
            print('=====================================================================================\n')

            # save model 
            self.model['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = copy.deepcopy(XgbTree)

            # save importance 
            if self.whether_importance:
                self.importance_df = self.xgb_importance(XgbTree)
                self.importance_dict['train{}_{} + test{}_{}'.format(
                    train_date_list[0], train_date_list[-1],
                    test_date_list[0], test_date_list[-1])
                ] = self.importance_df
            t4 = time.time()
            print(f"> Time: Train Model = {round(t4 - t3)}s\n")

        # else:
        #     train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
        #     test_orig_Y = self.DataTools.test_orig_Y
        #     train_orig_Y = self.DataTools.train_orig_Y
        #     train_pred_Y = train_orig_Y.copy(deep=True)
        #     # 准备非nan的mask
        #     na_idx = self.DataTools.na_idx
        #     if not cfg.dynamic_pool:
        #         self.y_mask = self.DataTools.test_mask
        #
        #     # 对label的值乘以100，避免数值精度影响模型结果
        #     if 'Regress' in self.type:
        #         train_Y = train_Y * 10
        #         test_Y = test_Y * 10
        #
        #     t3 = time.time()
        #     print('preparation is over.')
        #     print(f"> Time: Prepare Data = {round(t3 - t2)}s\n")
        #
        #     # 准备存储预测结果的dataframe
        #     pred_test_Y = test_Y.copy(deep=True)
        #
        #     with open(self.model_path + 'xgb_model_202108121716.pkl', 'rb') as f:
        #         XgbTree_dict = pickle.load(f)
        #         XgbTree = copy.deepcopy(XgbTree_dict['train20161227_20171220'])

        # XGBoost prediction
        print(f"Testing Xgb Model For {test_date_list[cfg.lag_date - 1]} to {test_date_list[-1]}")
        self.total_test_y.append(test_Y)
        if 'Class' in self.type:
            pred_test_y = np.array(XgbTree.predict_proba(test_xgb))[:, 1] # save probability as factors
            pred_test_class_y = np.array(XgbTree.predict(test_xgb)) # save corresponding classification results
            # 是否用动态票池加一层mask
            if cfg.diy_return:
                pred_test_class_y = pred_test_class_y * self.y_mask
                pred_test_y = pred_test_y * self.y_mask
        else:
            pred_test_y = np.array(XgbTree.predict(test_xgb))
            # 是否用动态票池加一层mask
            if cfg.diy_return:
                pred_test_y = pred_test_y * self.y_mask
        pred_test_Y.iloc[:] = pred_test_y
        self.total_test_period_df[test_date_list[cfg.lag_date - 1:]] = pred_test_Y.unstack()
        self.total_pred_test_y.append(pred_test_y)

        # out-of-sample
        test_y = test_Y.values.reshape(-1)
        idx = ~(np.isnan(test_y) | np.isinf(test_y) | np.isnan(pred_test_y) | np.isinf(pred_test_y))

        # plot
        pred_train_y = np.array(XgbTree.predict(train_xgb))
        train_pred_Y.iloc[na_idx[:, 0]] = pred_train_y
        train_y = train_Y.values
        print("====Explanatory Power====")
        self.plot_eps_fig(predy=pred_test_y[idx], testy=test_y[idx], date=train_date_list[-1], type='outsample')
        self.plot_eps_fig(predy=pred_train_y, testy=train_y, date=train_date_list[-1], type='insample')
        self.plot_rank_fig(predy=pred_test_Y.unstack(), testy=test_Y.unstack(), date=train_date_list[-1],
                           type='outsample')

        print("=====Evaluation Stat=====")
        if 'Regress' in self.type:
            try:
                r2 = R2(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                # ic = IC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                # smic = smIC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                ic = IC_cs(pred_y=pred_test_Y.unstack(), orig_y=test_orig_Y.unstack())
                smic = smIC_cs(pred_y=pred_test_Y.unstack(), orig_y=test_orig_Y.unstack())
                rmse = RMSE(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                self.result_dict['out_IC_list'].append(ic)
                self.result_dict['out_rankIC_list'].append(smic)
                self.result_dict['out_rmse_list'].append(rmse)
                print('OutSample R2 = {:.5f}, IC = {:.5f}, rankIC = {:.5f}, RMSE = {:.5f}'.format(r2, ic, smic, rmse))
                r2 = R2(pred_y=pred_train_y, orig_y=train_y)
                # ic = IC(pred_y=pred_train_y, orig_y=train_y)
                # smic = smIC(pred_y=pred_train_y, orig_y=train_y)
                ic = IC_cs(pred_y=train_pred_Y.unstack(), orig_y=train_orig_Y.unstack())
                smic = smIC_cs(pred_y=train_pred_Y.unstack(), orig_y=train_orig_Y.unstack())
                rmse = RMSE(pred_y=pred_train_y, orig_y=train_y)
                self.result_dict['in_IC_list'].append(ic)
                self.result_dict['in_rankIC_list'].append(smic)
                self.result_dict['in_rmse_list'].append(rmse)
                print('InSample R2 = {:.5f}, IC = {:.5f}, rankIC = {:.5f}, RMSE = {:.5f}'.format(r2, ic, smic, rmse))
            except:
                print('no valid test result.')
        elif 'Class' in self.type:
            try:
                error_arr = error_rate(pred_y=pred_test_class_y[idx], orig_y=test_y[idx])
                ic = IC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                smic = smIC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                self.result_dict['out_IC_list'].append(ic)
                self.result_dict['out_rankIC_list'].append(smic)
                print('Out Sample accuracy = {:.5f}, IC = {:.5f}, rankIC = {:.5f}'.format(error_arr[0], ic, smic))
                print('Out Sample precision = {:.5f}, recall = {:.5f}, f1_score = {:.5f}'.format(error_arr[1],
                                                                                                 error_arr[2],
                                                                                                 error_arr[3]))

            except:
                print('no valid test result.')

    # ———————————— plotting ——————————— #
    def plot_result_fig(self):
        plt.figure(figsize=(30, 6), dpi=300)
        p1, = plt.plot(self.result_dict['period_list'], self.result_dict['in_IC_list'], marker='o', color = 'orangered')
        p2, = plt.plot(self.result_dict['period_list'], self.result_dict['in_rankIC_list'], marker='o', color = 'orange')
        p3, = plt.plot(self.result_dict['period_list'], self.result_dict['in_rmse_list'], marker='o', color = 'limegreen')
        p4, = plt.plot(self.result_dict['period_list'], self.result_dict['out_IC_list'], marker='o', color = 'orangered')
        p5, = plt.plot(self.result_dict['period_list'], self.result_dict['out_rankIC_list'], marker='o', color = 'orange')
        p6, = plt.plot(self.result_dict['period_list'], self.result_dict['out_rmse_list'], marker='o', color = 'royalblue')
        l = len(self.result_dict['period_list'])
        plt.plot(self.result_dict['period_list'], [0.0] * l, linestyle='--', color='silver')
        plt.plot(self.result_dict['period_list'], [0.1] * l, linestyle = '--', color='silver')
        plt.plot(self.result_dict['period_list'], [0.2] * l, linestyle = '--', color='silver')
        plt.xticks(np.arange(len(self.result_dict['period_list'])), self.result_dict['period_list'], rotation = 45)
        plt.legend(handles=[p1, p2, p3, p4, p5, p6], labels=['IC', 'rankIC'], loc='best', fontsize=15)
        plt.title('xgb_result_plot')
        plt.savefig(self.fig_save_path + 'xgb_result_{}.png'.format(self.number))
        plt.close()

    def plot_eps_fig(self, predy, testy, date, type):
        plt.figure(figsize=(10, 10), dpi=200)
        draw_x = testy
        draw_y = predy - testy
        slope_out = slope(pred_y=draw_y, orig_y=draw_x)
        print('{} = {:.3f}%'.format(type, 100 * (slope_out + 1)))

        lim_max_p1 = np.max([np.abs(np.max(draw_x[np.where(draw_x > 0)])), np.abs(np.min(draw_x[np.where(draw_x > 0)])),
                             np.abs(np.max(draw_y[np.where(draw_x > 0)])), np.abs(np.min(draw_y[np.where(draw_x > 0)]))])
        lim_max_p2 = np.max([np.abs(np.max(draw_x[np.where(draw_x <= 0)])), np.abs(np.min(draw_x[np.where(draw_x <= 0)])),
                             np.abs(np.max(draw_y[np.where(draw_x <= 0)])), np.abs(np.min(draw_y[np.where(draw_x <= 0)]))])
        lim_max = max(lim_max_p1, lim_max_p2)
        plt.xlim(-lim_max, lim_max)
        plt.ylim(-lim_max, lim_max)
        p1 = plt.scatter(draw_x[np.where(draw_x > 0)],
                         draw_y[np.where(draw_x > 0)], s=2, color='lightcoral', alpha=0.3)
        p2 = plt.scatter(draw_x[np.where(draw_x <= 0)],
                         draw_y[np.where(draw_x <= 0)], s=2, color='royalblue', alpha=0.3)
        plt.xlabel('orig_y')
        plt.ylabel('pred_y')
        axis_x = np.linspace(-5, 5, 10)
        zero_line = np.linspace(0, 0, 10)
        neg_line = np.linspace(5, -5, 10)
        plt.plot(axis_x, zero_line, linestyle='--', color='darkgrey')
        plt.plot(axis_x, neg_line, linestyle='--', color='darkgrey')
        plt.legend(handles=[p1, p2], labels=['pos_stocks', 'neg_stocks'], loc='best',
                   fontsize=12)
        plt.title('xgb_scatter_plot_{}'.format(type))
        plt.savefig(self.fig_save_path + 'xgb_scatter_{}_{}_{}.png'.format(self.number, date, type))
        plt.close()

    def plot_rank_fig(self, predy, testy, date, type):
        plt.figure(figsize=(10, 10), dpi=200)
        for col in predy.columns:
            pred = predy[col].values
            test = testy[col].values
            col_idx = ~(np.isnan(pred) | np.isinf(pred) | np.isnan(test) | np.isinf(test))
            plt.scatter(testy[col].iloc[col_idx].rank().values, predy[col].iloc[col_idx].rank().values,
                        s=0.5, color='mediumvioletred', alpha=0.5)
        plt.xlabel('test_y')
        plt.ylabel('pred_y')
        plt.title('xgb_rank_plot_{}'.format(type))
        plt.savefig(self.fig_save_path + 'xgb_rank_{}_{}_{}.png'.format(self.number, date, type))
        plt.close()

    # ———————— custom loss function ———————— #
    def xgbreg_loss_function(self, pred_arr, dmatrix):
        real_arr = dmatrix.get_label()
        grad = 2 * (pred_arr - real_arr)
        hess = 2 * np.ones(len(pred_arr))
        return grad, hess

    def xgbreg_loss_function_quadrant(self, real_arr, pred_arr):
        high_penalty = real_arr ** real_arr + 2
        low_penalty = 1

        weight = np.where((real_arr >= 0), high_penalty, low_penalty)

        grad = 2 * (pred_arr - real_arr) * weight
        hess = 2 * weight

        return grad, hess

    def xgbreg_loss_function_golong(self, real_arr, pred_arr):
        def sigmoid(x, w, b):
            return 1 / (1 + np.exp(b - w * x))

        def p_huber(x):
            return np.log(np.exp(x) + np.exp(-x)) - np.log(2)

        crit = 2
        alpha = 0.7

        weight_pos = alpha * (1 - sigmoid(real_arr, 1, crit)) * p_huber(real_arr) + alpha * sigmoid(real_arr, 1, crit) * np.log(real_arr ** 2 + 1)
        weight_neg = (1 - alpha) * (1 - sigmoid(real_arr, 1, -crit)) * p_huber(real_arr) + (1 - alpha) * sigmoid(real_arr, 1, -crit) * np.log(real_arr ** 2 + 1)
        weight = np.where(real_arr >= 0, weight_pos, weight_neg)

        grad = 2 * (pred_arr - real_arr) * weight
        hess = 2 * weight

        return grad, hess

    def xgbreg_loss_function_huber(self, real_arr, pred_arr):
        d = pred_arr - real_arr
        h = 1  # h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess

    def xgbreg_loss_function_flower(self, real_arr, pred_arr):
        def sigmoid(x, w, b):
            return 1 / (1 + np.exp(b - w * x))

        def p_huber(x):
            return np.log(np.exp(x) + np.exp(-x)) - np.log(2)

        def tanh(x):
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        diff_arr = pred_arr - real_arr
        crit = 2

        def cal_element(diff):
            grad_pos = 1.4 * diff * sigmoid(diff, 1, crit) / (diff ** 2 + 1) + (sigmoid(diff, 1, crit) ** 2) * 0.7 * np.exp(crit - diff) * np.log(diff ** 2 + 1)
            grad_pos += 0.7 * (1 - sigmoid(diff, 1, crit)) * tanh(diff)
            grad_pos -= 0.7 * p_huber(diff) * (sigmoid(diff, 1, crit) ** 2) * np.exp(crit - diff)

            hess_pos = 0.7 * (1 - sigmoid(diff, 1, crit)) - 1.4 * (sigmoid(diff, 1, crit) ** 2) * tanh(diff) * np.exp(crit - diff)
            hess_pos -= (0.7 * (tanh(diff) ** 2) * (1 - sigmoid(diff, 1, crit)) + 2.8 * (diff ** 2) * sigmoid(diff, 1, crit) / ((diff ** 2 + 1) ** 2))
            hess_pos += (1.4 * sigmoid(diff, 1, crit) / (diff ** 2 + 1) + 2.8 * (sigmoid(diff, 1, crit) ** 2) * np.exp(crit - diff) * diff / (diff ** 2 + 1))
            hess_pos += (0.7 * np.exp(crit - diff) * (sigmoid(diff, 1, crit) ** 2) - 1.4 * np.exp(2 * crit - 2 * diff) * (sigmoid(diff, 1, crit) ** 3)) * (p_huber(diff) - np.log(diff ** 2 + 1))

            grad_neg = 0.6 * diff * sigmoid(diff, 1, -crit) / (diff ** 2 + 1) + (sigmoid(diff, 1, -crit) ** 2) * 0.3 * np.exp(-crit - diff) * np.log(diff ** 2 + 1)
            grad_neg += 0.3 * (1 - sigmoid(diff, 1, -crit)) * (np.exp(diff) - np.exp(-diff)) / (np.exp(diff) + np.exp(-diff))
            grad_neg -= 0.3 * p_huber(diff) * (sigmoid(diff, 1, -crit) ** 2) * np.exp(-crit - diff)

            hess_neg = 0.3 * (1 - sigmoid(diff, 1, -crit)) - 0.6 * (sigmoid(diff, 1, -crit) ** 2) * tanh(diff) * np.exp(-crit - diff)
            hess_neg -= (0.3 * (tanh(diff) ** 2) * (1 - sigmoid(diff, 1, -crit)) + 1.2 * (diff ** 2) * sigmoid(diff, 1, -crit) / ((diff ** 2 + 1) ** 2))
            hess_neg += (0.6 * sigmoid(diff, 1, -crit) / (diff ** 2 + 1) + 1.2 * (sigmoid(diff, 1, -crit) ** 2) * np.exp(-crit - diff) * diff / (diff ** 2 + 1))
            hess_neg += (0.3 * np.exp(-crit - diff) * (sigmoid(diff, 1, -crit) ** 2) - 0.6 * np.exp(-2 * crit - 2 * diff) * (sigmoid(diff, 1, -crit) ** 3)) * (p_huber(diff) - np.log(diff ** 2 + 1))

            grad = np.where(diff >= 0, grad_pos, grad_neg)
            hess = np.where(diff >= 0, hess_pos, hess_neg)

            return grad, np.where(hess < 0, 0, hess)

        grad_arr, hess_arr = cal_element(diff_arr)
        hess_arr = np.ones_like(hess_arr) * 2

        return grad_arr, hess_arr

    def xgb_feval_func(self, pred_arr, dmatrix):
        real_arr = dmatrix.get_label()
        rank_arr = np.argsort(np.argsort(pred_arr))
        group_ret = np.nanmean(np.where(rank_arr >= len(rank_arr) - 100, 1, np.nan) * real_arr)

        return 'eval-ret', group_ret

    def xgb_importance(self, model):
        # ————————模型重要性———————— #
        score1 = model.get_score(importance_type='gain')
        score2 = model.get_score(importance_type='weight')
        score3 = model.get_score(importance_type='cover')
        score4 = model.get_score(importance_type='total_gain')
        score5 = model.get_score(importance_type='total_cover')
        df = pd.DataFrame([score1, score2, score3, score4, score5], index=['gain', 'weight', 'cover', 'total_gain', 'total_cover']).T
        # 对于每一次看是否需要标准化
        # df = (df - df.mean()) / df.std()
        # df['TotalScore'] = df.mean(axis=1)
        # df.sort_values('TotalScore', inplace=True, ascending=False)
        return df



'''
---------------- LightGBM -------------------------
'''

class LgbModel:
    def __init__(self):
        self.train_mode = cfg.train_mode
        # datatools 
        self.DataTools = DataTools(predict_type=cfg.predict_type)

        # config 
        self.date_list = self.DataTools.date_list
        self.stock_pool = self.DataTools.stock_pool
        self.features = cfg.features
        self.train_period = cfg.train_period
        self.test_period = cfg.test_period
        self.train_pos = 0
        self.type = cfg.predict_type
        self.bisection = cfg.bisection
        self.log_path = cfg.log_path
        self.save_log = cfg.save_log
        self.model_path = cfg.model_path
        self.model_params_path = cfg.model_params_path
        self.whether_importance = cfg.whether_importance
        self.weight_method = cfg.weight_method
        self.test_delay = cfg.return_start_date - cfg.return_end_date
        self.model_record_path = cfg.model_record_path

    def myLgbModel(self, paras=None):
        """ set LGBM parameters """

        # default_paras = {
        #    'n_estimators': 150,
        #    'max_depth': 9,
        #    'learning_rate': 0.05,
        #    'subsample': 1,  # 随机采样训练样本 训练实例的子采样比
        #    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1-0.2
        #    'reg_lambda': 1,  # 控制模型复杂度的权重值的L2正则化项参数
        #    # 最大增量步长，我们允许每个树的权重估计。
        #    'max_delta_step': 0,
        #    'colsample_bytree': 1,  # 生成树时进行的列采样
        #    'booster': 'gbtree',
        #    'missing': np.nan,  # 自动处理nan
        #    'tree_method': 'gpu_hist'}

        default_paras = {
            'n_estimators': 50,
            'num_leaves': 30,
            'max_depth': 10,
            'learning_rate': 0.1,
            'min_child_samples': 20,
            # 'subsample' :1,
            # 'reg_lambda' :1,
            # 'colsample_bytree': 1,
            # 'boosting_type': 'gbdt',
            'n_jobs': -1
        }

        if paras is not None:
            for k in paras.keys():
                default_paras[k] = paras[k]

        if 'Regress' in self.type:
            default_paras['metric'] = 'rmse'
            default_paras['objective'] = 'regression'
            LgbTree = lgb.LGBMRegressor(**default_paras)
        elif 'Class' in self.type:
            default_paras['metric'] = 'auc'
            default_paras['objective'] = 'binary'
            LgbTree = lgb.LGBMClassifier(**default_paras)
        else:
            print("Wrong Type in LgBoost Training. Use Regressor.")
            default_paras['metric'] = 'rmse'
            default_paras['objective'] = 'regression'
            LgbTree = lgb.LGBMRegressor(**default_paras)

        return LgbTree

    def LgbTrain(self):
        print(f"LGB Training")
        print(f"Train Mode = {self.train_mode}")
        print(f"Train Type = {self.type}")
        print(f"Date to be trained: {self.date_list[0]} to {self.date_list[-1]}")
        print(f"Using Stock Pool = {cfg.fixed_pool}")
        print(f"Feature Num = {len(self.features)}")
        # get time to save models 
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M', time.localtime(time.time()))

        self.LgbDef = self.myLgbModel(paras=cfg.lgb_paras)
        self.number = curr_date + curr_time
        # with open(self.model_params_path + "lgb_{}.pkl".format(self.number), 'wb') as f:
        #    pickle.dump(self.LgbDef.get_params(), f)
        self.model = {}  # 用来存每次滚动训练的线性模型
        self.total_test_y = []  # 用来存每次训练魔性的原始label
        self.total_pred_test_y = []  # 用来存每次训练魔性的预测label
        self.importance_dict = {}  # 用来存每次模型特征重要性
        self.IC_list = []  # 用来存每次模型的IC
        self.rankIC_list = []  # 用来存每次模型的rankIC

        # store all labels as a dataframe index = stock_pool, columns = predict_period
        self.total_test_period_df = pd.DataFrame(
            index=self.stock_pool,
            columns=self.date_list[self.train_period:], 
            dtype='float'
        )

        self.fig_save_path = cfg.fig_path + 'lgb_fig_{}/'.format(self.number)
        if not os.path.isdir(self.fig_save_path):
            os.mkdir(self.fig_save_path)

        # 记录模型训练的各种信息
        # others = '' # 备注
        # record_df = pd.read_csv(self.model_record_path, index_col = 0)
        # if len(self.stock_pool) > 15:
        #     record_stock_pool = 'all'
        # else:
        #     record_stock_pool = self.stock_pool
        # record_info = [cfg.start_date, cfg.end_date, str(self.features), record_stock_pool, self.train_mode,
        #                 self.train_period, self.test_period, self.bisection, cfg.return_type, cfg.return_start_date,
        #                 cfg.return_end_date, self.weight_method, self.type, cfg.lag_date, int(cfg.valid_y), others]
        # record_df.loc[self.number] = record_info
        # record_df.to_csv(self.model_record_path)

        t_start = time.time()

        while (self.train_pos + self.train_period + self.test_delay + cfg.lag_date) <= len(self.date_list):
            t0 = time.time()

            # specify train test time frame
            if "roll" in self.train_mode:
                train_date_list = self.date_list[self.train_pos:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "expand" in self.train_mode:
                train_date_list = self.date_list[:self.train_pos + self.train_period + cfg.lag_date - 1]
                test_date_list = self.date_list[self.train_pos + self.train_period + self.test_delay:
                                                min(self.train_pos + self.train_period + self.test_period +
                                                    self.test_delay + cfg.lag_date - 1, len(self.date_list))]
            elif "bisect" in self.train_mode:
                train_date_list = self.date_list[:int(len(self.date_list) * self.bisection)]
                test_date_list = self.date_list[int(len(self.date_list) * self.bisection) + self.test_delay:]
            else:
                print("...Error:{}...".format(self.train_mode))
                break
            print(f"\nTraining Lgb Model For {train_date_list[0]} to {train_date_list[-1]}")

            # parepare data
            self.DataTools.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
            t2 = time.time()
            print(f"Time: Load Data = {t2 - t0}")
            print('preparation is over.')
            if 'Class' in self.type:
                train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
                test_orig_Y = self.DataTools.test_orig_Y
                train_orig_Y = self.DataTools.train_orig_Y_mask
            else:
                train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
                train_orig_Y = self.DataTools.train_orig_Y_mask

            # multiply by 100 for numerical precision
            if 'Regress' in self.type:
                train_Y = train_Y * 100
                test_Y = test_Y * 100
            print(f'Mean Train Y = {np.nanmean(train_Y.values)}')
            print(f'Train X Shape:{train_X.shape}')

            # parepare to be saved s
            pred_test_Y = test_Y.copy(deep=True)

            # sample weights
            weights = self.DataTools.weights_cal(train_Y, self.weight_method)

            # Train 

            # 滚动调参模块
            if cfg.lgb_rolling_paras:
                raise NotImplementedError('LGBM Para Search Terminated')
                if 'Regress' in self.type:
                    Lgb_Para = lgb.LGBMRegressor(n_jobs=50)
                    # else:
                    #    print('...Error:{}'.format(self.model_type))
                    #    break
                elif 'Class' in self.type:
                    Lgb_Para = lgb.LGBMClassifier(n_jobs=50)

                if cfg.self_defined_score == True:
                    score_function = make_scorer(eval(cfg.score_metric))
                else:
                    score_function = cfg.score_metric

                t0 = time.time()

                if 'Regress' in self.type:
                    grid = GridSearchCV(Lgb_Para, param_grid=cfg.lgb_paras_test, scoring=score_function, cv=2,
                                        refit=False).fit(train_X, train_Y, sample_weight=weights)
                    print(f"Best parameters = {grid.best_params_}")
                    LgbTree = self.myLgbModel(paras=grid.best_params_)

                elif 'Class' in self.type:
                    if 'best_group_return' in cfg.score_metric:
                        kf = KFold(n_splits=2)
                        CV_record = []
                        params_list = []
                        for p in [cfg.lgb_paras_test]:
                            # Always sort the keys of a dictionary, for reproducibility
                            items = sorted(p.items())
                            keys, values = zip(*items)
                            for v in product(*values):
                                params = dict(zip(keys, v))
                                params_list.append(params)

                                ret = 0
                                # For each set of params, fit on CV set
                                for train, test in kf.split(train_orig_Y):
                                    Train_X_k = train_X.iloc[train,]
                                    Train_Y_k = train_Y[train]
                                    # Train_orig_Y_k = Train_orig_Y[train]
                                    Test_X_k = train_X.iloc[test,]
                                    # Test_Y_k = Train_Y[test]
                                    Test_orig_Y_k = train_orig_Y[test]

                                    Lgb_Para = lgb.LGBMClassifier(**params)
                                    Lgb_Para.fit(Train_X_k, Train_Y_k)
                                    pred_test_y = Lgb_Para.predict_proba(Test_X_k)[:, 1]
                                    r = best_group_return(Test_orig_Y_k, pred_test_y)
                                    ret = ret + r
                                CV_record.append(ret)
                        print(f"Best parameters = {params_list[np.argmax(CV_record)]}")
                        LgbTree = self.myLgbModel(paras=params_list[np.argmax(CV_record)])

                    else:
                        grid = GridSearchCV(Lgb_Para, param_grid=cfg.lgb_paras_test, scoring=score_function, cv=2,
                                            refit=False).fit(train_X, train_Y, sample_weight=weights)
                        print(f"Best parameters = {grid.best_params_}")
                        LgbTree = self.myLgbModel(paras=grid.best_params_)

                # self.LgbDef = self.myLgbModel(paras = grid.best_params_)
                # LgbTree = self.myLgbModel(paras=grid.best_params_)

            else:
                LgbTree = self.LgbDef

            # if cfg.record_loss:
            #     idx = ~(np.isnan(test_Y) | np.isinf(test_Y))
            #     LgbTree.fit(
            #         train_X, train_Y, sample_weight=weights,
            #         eval_set=[(train_X.iloc[-2000000:], train_Y[-2000000:]), (test_X.iloc[idx], test_Y[idx])],
            #         # eval_metric = ['error'],
            #         verbose=False
            #     )
            # else:
            LgbTree.fit(train_X, train_Y, sample_weight=weights)

            # save model
            self.model['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = copy.deepcopy(LgbTree)

            # save feature importance
            if self.whether_importance:
                self.importance_df = self.lgb_importance(LgbTree)
                self.importance_dict['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = self.importance_df

            # 滚动变量选择模块：可重新用筛选的变量fit模型
            if cfg.lgb_rolling_feature_selection:
                raise NotImplementedError('LGBM Para Search Terminated')
                train_X_new = train_X.loc[:, self.importance_df[:cfg.lgb_num_feature].index]
                test_X_new = test_X.loc[:, self.importance_df[:cfg.lgb_num_feature].index]
                print(f'new Train X Shape:{train_X_new.shape}')
                LgbTree.fit(train_X_new, train_Y, sample_weight=weights)
                # print(LgbTree.booster_.feature_importance(importance_type='gain').shape[0])

                # 使用新的LgBoost模型进行预测
                print(f"Testing Lgb Model For {test_date_list[cfg.lag_date - 1]} to {test_date_list[-1]}")
                self.total_test_y.append(test_Y)
                if 'Class' in self.type:
                    pred_test_y = LgbTree.predict_proba(test_X_new)[:, 1]
                    pred_test_class_y = LgbTree.predict(test_X_new)
                else:
                    pred_test_y = LgbTree.predict(test_X_new)
                pred_test_Y.iloc[:] = pred_test_y
                self.total_test_period_df[test_date_list[cfg.lag_date - 1:]] = pred_test_Y.unstack()
                self.total_pred_test_y.append(pred_test_y)
            else:
                # predict
                print(f"Testing Lgb Model For {test_date_list[cfg.lag_date - 1]} to {test_date_list[-1]}")
                self.total_test_y.append(test_Y)
                if 'Class' in self.type:
                    pred_test_y = LgbTree.predict_proba(test_X)[:, 1]
                    pred_test_class_y = LgbTree.predict(test_X)
                else:
                    pred_test_y = LgbTree.predict(test_X)
                pred_test_Y.iloc[:] = pred_test_y
                self.total_test_period_df[test_date_list[cfg.lag_date - 1:]] = pred_test_Y.unstack()
                self.total_pred_test_y.append(pred_test_y)

            # out-of-sample
            test_y = test_Y.values.reshape(-1)
            idx = ~(np.isnan(test_y) | np.isinf(test_y) | np.isnan(pred_test_y) | np.isinf(pred_test_y))

            # 临时绘图
            if cfg.lgb_rolling_feature_selection:
                raise NotImplementedError('LGBM Para Search Terminated')
                pred_train_y = LgbTree.predict(train_X_new)
            else:
                pred_train_y = LgbTree.predict(train_X)
            train_y = train_Y.values
            print("====Explanatory Power====")
            self.plot_eps_fig(predy=pred_test_y[idx], testy=test_y[idx], date=train_date_list[-1], type='outsample')
            self.plot_eps_fig(predy=pred_train_y, testy=train_y, date=train_date_list[-1], type='insample')
            self.plot_rank_fig(predy=pred_test_Y.unstack(), testy=test_Y.unstack(), date=train_date_list[-1],
                               type='outsample')

            print("====Evaluation Stats====")
            if 'Regress' in self.type:
                try:
                    r2 = R2(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                    ic = IC(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                    smic = smIC(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                    rmse = RMSE(pred_y=pred_test_y[idx], orig_y=test_y[idx])
                    self.IC_list.append(ic)
                    self.rankIC_list.append(smic)
                    print('Out Sample R2 = {:.5f}, IC = {:.5f}, rankIC = {:.5f}, RMSE = {:.5f}'.format(r2, ic, smic,
                                                                                                       rmse))
                except:
                    print('no valid test result.')
            elif 'Class' in self.type:
                try:
                    error_arr = error_rate(pred_y=pred_test_class_y[idx], orig_y=test_y[idx])
                    ic = IC(pred_y=pred_test_y[idx], orig_y=test_orig_Y[idx])
                    smic = smIC(pred_y=pred_test_y[idx], orig_y=test_orig_Y[idx])
                    self.IC_list.append(ic)
                    self.rankIC_list.append(smic)
                    print('Out Sample accuracy = {:.5f}, IC = {:.5f}, rankIC = {:.5f}'.format(error_arr[0], ic, smic))
                    print('Out Sample precision = {:.5f}, recall = {:.5f}, f1_score = {:.5f}'.format(error_arr[1],
                                                                                                     error_arr[2],
                                                                                                     error_arr[3]))

                except:
                    print('no valid test result.')

            t1 = time.time()
            print(f"Using Time = {t1 - t0}")
            self.train_pos += self.test_period

            if 'bisect' in self.train_mode:
                break

        # 备用代码：将数据去除极值
        # df['pred_test_y'] = df['pred_test_y'].clip(lower=-30, upper=30)
        # df.to_csv(log_dir + '/y_recorder.csv', index=False)

        # save 
        with open(os.path.join(self.model_path, "lgb_{}.pkl".format(self.number)), 'wb') as f:
            pickle.dump(self.model, f)

        t_end = time.time()
        print(f"Total IC = {np.nanmean(np.array(self.IC_list))}")
        print(f"Total rank IC = {np.nanmean(np.array(self.rankIC_list))}")
        print(f"Time: Total training time = {t_end - t_start}")

        # 将预测结果输出成因子 h5 格式
        self.factor_name = "lgb_{}_{}_{}_{}_{}_{}".format(self.date_list[self.train_period][2:], self.date_list[-1][2:],
                                                          self.train_period, self.test_period, self.number, self.type)
        self.DataTools.save_factor_data(feature_name=self.factor_name, eod_df=self.total_test_period_df,
                                        factor_type="ml_factors")

        # 将模型重要性储存
        print(f'Trained Factor Name is {self.factor_name}')
        self.importance_dict['factor_list'] = self.features
        with open(os.path.join(self.log_path, 'lgb_importance_{}.pkl'.format(self.number)), 'wb') as f:
            pickle.dump(self.importance_dict, f)
        # 
        return self.total_test_y, self.total_pred_test_y

    # ———————————— plotting ———————————— #
    def plot_eps_fig(self, predy, testy, date, type):
        plt.figure(figsize=(10, 10), dpi=200)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        draw_x = testy
        draw_y = predy - testy
        slope_out = slope(pred_y=draw_y, orig_y=draw_x)
        print('{} = {:.3f}%'.format(type, 100 * (slope_out + 1)))
        p1 = plt.scatter(draw_x[np.where(draw_x > 0)],
                         draw_y[np.where(draw_x > 0)], s=2, color='lightcoral', alpha=0.3)
        p2 = plt.scatter(draw_x[np.where(draw_x <= 0)],
                         draw_y[np.where(draw_x <= 0)], s=2, color='royalblue', alpha=0.3)
        axis_x = np.linspace(-5, 5, 10)
        zero_line = np.linspace(0, 0, 10)
        neg_line = np.linspace(1, -1, 10)
        plt.plot(axis_x, zero_line, linestyle='--', color='darkgrey')
        plt.plot(axis_x, neg_line, linestyle='--', color='darkgrey')
        plt.legend(handles=[p1, p2], labels=['pos_stocks', 'neg_stocks'], loc='best',
                   fontsize=12)
        plt.title('lgb_scatter_plot_{}'.format(type))
        plt.savefig(self.fig_save_path + 'lgb_scatter_{}_{}_{}.png'.format(self.number, date, type))
        plt.close()

    def plot_rank_fig(self, predy, testy, date, type):
        plt.figure(figsize=(10, 10), dpi=200)
        for col in predy.columns:
            pred = predy[col].values
            test = testy[col].values
            col_idx = ~(np.isnan(pred) | np.isinf(pred) | np.isnan(test) | np.isinf(test))
            plt.scatter(testy[col].iloc[col_idx].rank().values, predy[col].iloc[col_idx].rank().values,
                        s=0.5, color='mediumvioletred', alpha=0.5)
        plt.title('lgb_rank_plot_{}'.format(type))
        plt.savefig(self.fig_save_path + 'lgb_rank_{}_{}_{}.png'.format(self.number, date, type))
        plt.close()

    def lgb_importance(self, model):
        """ feature importance """
        #    score1 = model.get_booster().get_score(importance_type='gain')
        #    score2 = model.get_booster().get_score(importance_type='weight')
        #    score3 = model.get_booster().get_score(importance_type='cover')

        score1 = model.booster_.feature_importance(importance_type='gain')
        score2 = model.booster_.feature_importance(importance_type='split')
        df = pd.DataFrame([score1, score2], index=['gain', 'split']).T
        # 对于每一次看是否需要标准化
        # df = (df - df.mean()) / df.std()
        df['TotalScore'] = df.mean(axis=1)
        df.index = self.DataTools.features
        df.sort_values('TotalScore', inplace=True, ascending=False)
        return df

    # def Lgb_Para_Search(self):
    #     # ————————LightGBM模型调参———————— #
    #     print(f"Lgb Parameter Search by Cross Validation")
    #     print(f"Train Type = {self.type}")
    #     print(f"Metric = {cfg.score_metric}")
    #     print(f"Date to be trained: {self.date_list[0]} to {self.date_list[-1]}")
    #     print(f"Using Stock Pool = {cfg.fixed_pool}")
    #     print(f"Feature Num = {len(self.features)}")

    #     train_date_list = self.date_list
    #     test_date_list = train_date_list

    #     # 获取现在的时间，用来记录模型编号
    #     curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
    #     curr_time = time.strftime('%H%M', time.localtime(time.time()))
    #     self.number = curr_date + curr_time

    #     self.DataTools.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
    #     print('preparation is over.')

    #     if 'Class' in self.type:
    #         train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
    #         test_orig_Y = self.DataTools.test_orig_Y
    #         train_orig_Y = self.DataTools.train_orig_Y_mask
    #     else:
    #         train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()
    #         train_orig_Y = self.DataTools.train_orig_Y_mask

    #     # 对label的值乘以100，避免数值精度影响模型结果
    #     if 'Regress' in self.type:
    #         train_Y = train_Y * 100
    #         test_Y = test_Y * 100
    #     print(f'Mean Train Y = {np.nanmean(train_Y.values)}')
    #     print(f'Train X Shape:{train_X.shape}')

    #     # 确定训练的sample_weights, weight_method在config中设置，'EW'即default等权
    #     weights = self.DataTools.weights_cal(train_Y, self.weight_method)

    #     if 'Regress' in self.type:
    #         Lgb_Para = lgb.LGBMRegressor(n_jobs=50)
    #         # else:
    #         #    print('...Error:{}'.format(self.model_type))
    #         #    break

    #     elif 'Class' in self.type:
    #         Lgb_Para = lgb.LGBMClassifier(n_jobs=50)

    #     if cfg.self_defined_score == True:
    #         score_function = make_scorer(eval(cfg.score_metric))
    #     else:
    #         score_function = cfg.score_metric

    #     t0 = time.time()

    #     if 'Regress' in self.type:
    #         grid = GridSearchCV(Lgb_Para, param_grid=cfg.lgb_paras_test, scoring=score_function, cv=2, refit=False).fit(
    #             train_X, train_Y, sample_weight=weights)
    #         print(f"Best parameters = {grid.best_params_}")
    #     elif 'Class' in self.type:
    #         if 'best_group_return' in cfg.score_metric:
    #             kf = KFold(n_splits=2)
    #             CV_record = []
    #             params_list = []
    #             for p in [cfg.lgb_paras_test]:
    #                 # Always sort the keys of a dictionary, for reproducibility
    #                 items = sorted(p.items())
    #                 keys, values = zip(*items)
    #                 for v in product(*values):
    #                     params = dict(zip(keys, v))
    #                     params_list.append(params)

    #                     ret = 0
    #                     # For each set of params, fit on CV set
    #                     for train, test in kf.split(train_orig_Y):
    #                         Train_X_k = train_X.iloc[train,]
    #                         Train_Y_k = train_Y[train]
    #                         # Train_orig_Y_k = Train_orig_Y[train]
    #                         Test_X_k = train_X.iloc[test,]
    #                         # Test_Y_k = Train_Y[test]
    #                         Test_orig_Y_k = train_orig_Y[test]

    #                         Lgb_Para = lgb.LGBMClassifier(**params)
    #                         Lgb_Para.fit(Train_X_k, Train_Y_k)
    #                         pred_test_y = Lgb_Para.predict_proba(Test_X_k)[:, 1]
    #                         r = best_group_return(Test_orig_Y_k, pred_test_y)
    #                         ret = ret + r
    #                     CV_record.append(ret)
    #             print(f"Best parameters = {params_list[np.argmax(CV_record)]}")

    #         else:
    #             grid = GridSearchCV(Lgb_Para, param_grid=cfg.lgb_paras_test, scoring=score_function, cv=2,
    #                                 refit=False).fit(train_X, train_Y, sample_weight=weights)
    #             print(f"Best parameters = {grid.best_params_}")

    #     # 得到CV结果
    #     # print(f"CV result = {grid.cv_results_['mean_test_score']}")
    #     if self.save_log:
    #         if ('Class' in self.type) & ('best_group_return' in cfg.score_metric):
    #             self.cv_result_df = pd.DataFrame({'params': params_list, 'Test Score': CV_record})
    #         else:
    #             self.cv_result_df = pd.DataFrame(
    #                 {'params': grid.cv_results_['params'], 'Test Score': grid.cv_results_['mean_test_score']})
    #         with open(self.log_path + 'lgb_CV_{}_{}_{}_{}.pkl'.format(self.date_list[0][2:], self.date_list[-1][2:],
    #                                                                   self.number, self.type), 'wb') as f:
    #             pickle.dump(self.cv_result_df, f)

    #     #  if 'Regress' in self.type:
    #     #      self.coef_df['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = coeff
    #     #  elif 'Class' in self.type:
    #     #      self.coef_df['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = coeff[0]

    #     t1 = time.time()
    #     print(f"Using Time = {t1 - t0}")

    #     return self.cv_result_df

'''
---------------- RandomForest -------------------------
'''

# TODO: fix RF 

class RandomForest:
    def __init__(self, offline=False, eod_data_dict=None):
        raise NotImplementedError('RF to be fixed')
        self.train_mode = cfg.train_mode
        self.type = cfg.predict_type
        self.offline = offline
        if self.offline:
            self.DataTools = DataTools(offline=self.offline, eod_data_dict=eod_data_dict, predict_type=self.type)
        else:
            self.DataTools = DataTools(offline=self.offline, predict_type=self.type)

        self.date_list = self.DataTools.date_list
        self.stock_pool = self.DataTools.stock_pool
        self.features = cfg.features
        self.gene_features = self.DataTools.gene_features
        self.train_period = cfg.train_period
        self.test_period = cfg.test_period
        self.weight_method = cfg.weight_method
        self.train_pos = 0
        self.bisection = cfg.bisection
        self.log_path = cfg.log_path
        self.save_log = cfg.save_log
        self.model_path = cfg.model_path

    def RFTrain(self):
        # ————————随机森林训练主体———————— #
        default_para = cfg.rf_paras
        print(f"Random Forest Training")
        print(f"Train Mode = {self.train_mode}")
        print(f"Date to be trained: {self.date_list[0]} to {self.date_list[-1]}")
        print(f"Using Stock Pool = {cfg.fixed_pool}")
        print(f"Feature Num = {len(self.features) + len(self.gene_features)}")
        # 获取现在的时间，用来记录模型编号
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M', time.localtime(time.time()))

        self.number = curr_date + curr_time
        self.model = {} # 用来存每次滚动训练的线性模型
        self.total_test_y = [] # 用来存每次训练魔性的原始label
        self.total_pred_test_y = [] # 用来存每次训练魔性的预测label

        # 用来储存feature importance
        self.feat_imp_dict = {}

        # 用来存整个预测区间的预测label(作为因子值)的dataframe, index = stock_pool, columns = predict_period
        self.total_test_period_df = pd.DataFrame(index=self.stock_pool, columns=self.date_list[self.train_period:], dtype='float')



        while (self.train_pos + self.train_period) <= len(self.date_list):
            t0 = time.time()

            # 按照训练模式设定每次滚动的训练时间区间和预测时间区间
            if "roll" in self.train_mode:
                train_date_list = self.date_list[self.train_pos:self.train_pos + self.train_period]
                test_date_list = self.date_list[self.train_pos + self.train_period:
                                                min(self.train_pos + self.train_period + self.test_period,
                                                    len(self.date_list))]
            elif "expand" in self.train_mode:
                train_date_list = self.date_list[:self.train_pos + self.train_period]
                test_date_list = self.date_list[self.train_pos + self.train_period:
                                                min(self.train_pos + self.train_period + self.test_period,
                                                    len(self.date_list))]
            elif "bisect" in self.train_mode:
                train_date_list = self.date_list[self.train_period:int(len(self.date_list) * self.bisection)]
                test_date_list = train_date_list
                # test_date_list = self.date_list[int(len(self.date_list) * self.bisection):]
            else:
                print("...Error:{}...".format(self.train_mode))
                break
            print(f"\nTraining Random Forest Model For {train_date_list[0]} to {train_date_list[-1]}")

            # 准备数据并且做清晰
            self.DataTools.prepare_data(train_date_list=train_date_list, test_date_list=test_date_list)
            print('preparation is over.')
            train_X, train_Y, test_X, test_Y = self.DataTools.standard_clean_data()

            # 对label的值乘以100，避免数值精度影响模型结果
            if "Regress" in self.type:
                train_Y = train_Y * 100
                test_Y = test_Y * 100
            print(f'Mean Train Y = {np.nanmean(train_Y.values)}')
            print(f'Train X Shape:{train_X.shape}, Train Y Shape:{train_Y.shape}')

            # 准备存储预测结果的dataframe
            pred_test_Y = test_Y.copy()

            # 确定训练的sample_weights, weight_method在config中设置，'EW'即default等权
            weights = self.DataTools.weights_cal(train_Y, self.weight_method, X = train_X,
                                           EX_X = None, features = cfg.features)

            # 训练线性模型
            if 'Class' in self.type:
                RF = RandomForestClassifier(**default_para)
            elif 'Regress' in self.type:
                RF = RandomForestRegressor(**default_para)
            else:
                print('...Error:{}'.format(self.type))
                break
            RF.feature_names = self.features 
            RF.fit(train_X.values, train_Y.values, sample_weight=weights) 
            # 备用：前向变量逐步回归代码
            # RF, feature_used = self.linear_forward_select(pd.concat([train_X, train_Y], axis = 1), 'label')

            # 提取feature importance
            if self.save_log:
                features = train_X.columns.values
                feat_imp = RF.feature_importances_
                self.feat_imp_dict['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = dict(zip(features, feat_imp))

            # 储存模型
            self.model['train{}_{}'.format(train_date_list[0], train_date_list[-1])] = copy.deepcopy(RF)
            with open(os.path.join(self.model_path, 'random_forest_model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)

            # 使用线性模型进行预测
            print(f"Testing Random Forest Model For {test_date_list[0]} to {test_date_list[-1]}")
            self.total_test_y.append(test_Y)
            if 'Class' in self.type:
                pred_test_y = RF.predict_proba(test_X)[:, 1]
                pred_test_class_y = RF.predict(test_X)
            else:
                pred_test_y = RF.predict(test_X)
            pred_test_Y.iloc[:] = pred_test_y
            self.total_test_period_df[test_date_list[cfg.lag_date - 1:]] = pred_test_Y.unstack()
            self.total_pred_test_y.append(pred_test_y)


            # 计算样本外的指标
            test_Y = test_Y.values.reshape(-1)
            idx = ~(np.isnan(test_Y) | np.isinf(test_Y) | np.isnan(pred_test_y) | np.isinf(pred_test_y))
            if 'Regress' in self.type:
                try:
                    r2 = R2(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    ic = IC(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    rmse = RMSE(pred_y=pred_test_y[idx], orig_y=test_Y[idx])
                    print(f'Out Sample R2 = {r2}, IC = {ic}, RMSE = {rmse}')
                except:
                    print('no valid test result.')
            elif 'Class' in self.type:
                try:
                    error_arr = error_rate(pred_y=pred_test_class_y[idx], orig_y=test_Y[idx])
                    print(f'Out Sample accuracy = {error_arr[0]}, precision = {error_arr[1]}, '
                          f'recall = {error_arr[2]}, f1_score = {error_arr[3]}')
                except:
                    print('no valid test result.')


            t1 = time.time()
            print(f"Using Time = {t1 - t0}")
            self.train_pos += self.test_period

            if 'bisect' in self.train_mode:
                break

        # 将预测结果输出成因子 h5 格式
        self.factor_name = "rf_bm_{}_{}_{}_{}_{}_{}".format(self.date_list[self.train_period][2:], self.date_list[-1][2:],
                                                      self.train_period, self.test_period, self.number, self.type)
        self.DataTools.save_factor_data(feature_name = self.factor_name, eod_df = self.total_test_period_df,
                                     factor_type = "ml_factors")
        print(f'Trained Factor Name is {self.factor_name}')

        # 将变量p值存到log
        if self.save_log:
            feat_imp_df = pd.DataFrame(self.feat_imp_dict).T
            feat_imp_df.to_csv(self.log_path + f'random_forest_feat_imp_{self.number}.csv')

        return self.total_test_y, self.total_pred_test_y
