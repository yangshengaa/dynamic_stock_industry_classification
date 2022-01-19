"""
Measure similarities (i.e. correlation) between stocks in a given time period.
"""

# load packages 
import numpy as np
import pandas as pd 

MIN_PERIODS = 60  # at least 2 months to start computing correlation

# =================================
# -------- similarities -----------
# =================================

def cor(return_df: pd.DataFrame) -> pd.DataFrame:
    """ correlation """
    corr_df = return_df.T.corr(method='pearson', min_periods=MIN_PERIODS)
    return corr_df

def rank_corr(return_df: pd.DataFrame) -> pd.DataFrame:
    """ spearman correlation """
    corr_df = return_df.T.corr(method='spearman', min_periods=MIN_PERIODS)
    return corr_df 

# TODO: coskew, cokurt (higher dimensional variance)
