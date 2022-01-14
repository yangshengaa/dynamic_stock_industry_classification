"""
辅助函数
"""

# load package 
import os
import pandas as pd 

# load files 
my_factor_path = 'data/features/factor'
my_support_factor_path = 'data/features/support_factor'

def read_eod_feature(factor_name: str, is_support_factor: bool) -> pd.DataFrame:
    """
    facilitate reading support factor (local support, without initializing the data server)
    """
    # get path 
    factor_path = os.path.join(
        my_support_factor_path if is_support_factor else my_factor_path,
        f'eod_{factor_name}'
    )
    # retrieve 
    factor_df = pd.read_feather(factor_path).set_index('index')
    return factor_df
