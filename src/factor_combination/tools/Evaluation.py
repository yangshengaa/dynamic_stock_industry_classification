"""
Model Evaluation
"""

# load packages 
import os
import copy
import numpy as np
import pandas as pd

from sklearn import linear_model
from scipy.stats import rankdata

# ==========================
# methods for rating models 
# ==========================

def corr_with(A,B):
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:,None], ssB[None]))

def R2(pred_y, orig_y):
    """ linear regression R2 """
    SYY = np.sum((orig_y - np.mean(orig_y)) ** 2)
    SSR = np.sum((orig_y - pred_y) ** 2)
    return 1 - (SSR / SYY)

def adjR2(pred_y, orig_y, p):
    """ adjusted linear regression R_squareed """
    SYY = np.sum((orig_y - np.mean(orig_y)) ** 2)
    SSR = np.sum((orig_y - pred_y) ** 2)
    R2 = 1 - SSR / SYY
    n = len(pred_y)
    return 1 - (1 - R2) * (n - 1) / (n - p - 1)

def IC(pred_y, orig_y):
    """ prediction and real y value correlation """
    # stack_pred_y = pred_y.reshape(period, len(pred_y) / period)
    # stack_orig_y = orig_y.reshape(period, len(orig_y) / period)
    # # 预测结果相关系数
    # return np.mean(np.diag(corr_with(stack_pred_y, stack_orig_y)))
    return np.corrcoef(pred_y, orig_y)[0, 1]

def smIC(pred_y, orig_y):
    """ prediction and real y value spearman correlation """
    # stack_pred_y = pred_y.reshape(period, len(pred_y) / period)
    # stack_orig_y = np.argsort(np.argsort(orig_y.reshape(period, len(orig_y) / period)))
    # 预测spearman相关系数
    return np.corrcoef(pred_y, np.argsort(np.argsort(orig_y)))[0, 1]

def IC_cs(pred_y, orig_y):
    """ mean of cross-sectional correlation """
    return pred_y.corrwith(orig_y).mean()

def smIC_cs(pred_y, orig_y):
    """ mean of cross-sectional spearman correlation """
    orig_rank_y = orig_y.rank(axis = 0)
    return pred_y.corrwith(orig_rank_y).mean()

def RMSE(pred_y, orig_y):
    """ prediction RMSE """
    return np.sqrt(np.sum((pred_y - orig_y) ** 2) / len(pred_y))

def best_group_return(orig_y, pred_y):
    """ best 10% averae return """
    # 最大10%预测y值(pred_y)对应的实际收益率(orig_y)平均
    rankY = rankdata(pred_y) / len(pred_y)
    bigY = (rankY > 0.9)
    r = np.mean(orig_y[np.where(bigY)[0]])
    return r

def error_rate(pred_y, orig_y):
    """ [acc, precision, recall, f1] for classification """
    # 准确率、查准率、查全率、f1score
    accuracy = np.sum(pred_y == orig_y) / len(pred_y)
    precision = np.dot(pred_y, orig_y) / np.sum(pred_y)
    recall = np.dot(pred_y, orig_y) / np.sum(orig_y)
    f1 = 2 * precision * recall / (precision + recall)
    return [accuracy, precision, recall, f1]

def slope(pred_y, orig_y):
    """ the slope of single variable linear regression """
    # 解释度（一元回归的系数）
    k = np.cov(pred_y, orig_y)[0, 1] / np.var(orig_y)
    return k
