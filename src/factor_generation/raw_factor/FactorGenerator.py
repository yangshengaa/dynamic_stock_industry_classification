"""
main function to generate factors 
"""
# load packages
import os 
import sys
import time
import getpass
import datetime
import warnings
import traceback

import numpy as np
import pandas as pd

import multiprocessing as mp 
from multiprocessing.shared_memory import SharedMemory

from typing import List, Dict

warnings.filterwarnings('ignore')
# sys.path.append('..')
# print(sys.path)
# user = getpass.getuser()  # otherwise filename too long
user = ''

# load file
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
import config as config
import factor_gen as fg
import support_factor_gen as sfg

# data source
# PqiDataSdk offline 
sys.path.append(os.path.join(cur_dir, '../..'))
from data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
ds = PqiDataSdkOffline()

# from PqiDataSdk import * 
# ds = PqiDataSdk(user=user, size=40, pool_type='mp')
# ds_single = PqiDataSdk(user=user, size=1, pool_type='mp')  # 辅助trade因子的多进程

# save constants 
concat = pd.concat

class FactorGenerator:

    def __init__(self):
        """
        初始化（转移设置）
        """
        # 转移设置
        self.factor_path = config.my_factor_path
        self.support_factor_path = config.my_support_factor_path
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.tickers = config.tickers
        self.is_support_factor = config.is_support_factor
        self.required_data_types = config.required_data_types
        self.required_field_types_dict = config.required_field_types_dict
        self.index_mins_data_dict = None  # 新建一个空的，有用再说

        # 计算因子路径
        self.des = 'support_factor' if self.is_support_factor else 'factor'

        # 选择因子列表
        self.source = 'sfg' if self.is_support_factor else 'fg'
        self.name_list = config.support_factor_name_list if self.is_support_factor else config.factor_name_list
        self.param_lists = config.support_factor_params if self.is_support_factor else config.factor_params        



    # ----------------------------------------------------------------
    # ------------------------- 不同类别的数据获取 --------------------
    # ---------------------------------------------------------------

    # # ------ 无需读取任何数据 （只读support factors的）----------------
    # def process_each_read_only_factor(self, factor, param_list):
    #     """
    #     子进程，辅助生成只读取support_factor的因子
    #     :param factor: 因子名称，同时也是对应的函数名称
    #     :param param_list: 对应指定的因子参数
    #     """
    #     # 生成因子名
    #     start = time.time()
    #     factor_name = '_'.join([factor] + [str(x) for x in param_list])
    #     # print(f'计算因子{factor_name}')
    #     dummy_stock_data_dict = {}
    #     # 计算因子
    #     factor_df = eval(f'{self.source}.{factor}(dummy_stock_data_dict, param_list)')
    #     factor_df = factor_df + factor_df * 0 
    #     # 储存因子
    #     parent_path = self.support_factor_path 
    #     ds.save_eod_feature(factor_name, factor_df, self.parent_path)
    #     print(f'完成因子{factor_name}存储, 耗时{time.time() - start}')


    # def run_no_required_data(self):
    #     """ 
    #     对于无需从dataserver获取任何数据的因子（只读入support factor的因子）
    #     """
    #     pool = mp.Pool(processes=8)
    #     process_list = []
    #     for factor in self.name_list:
    #         factor_param_list = self.param_lists[factor]
    #         for param_list in factor_param_list:
    #             process_list.append(pool.apply_async(self.process_each_read_only_factor, args=(factor, param_list)))
                
    #     pool.close()
    #     pool.join()
        
    
    # # ---------------------- 多进程结果回传辅助函数 -------------------------

    # def init_support_factor_agg_dict(self):
    #     """
    #     给trade, depth, mins, 5_mins, 3to1创建空的support factor agg dict, 结构为单层dict
    #     {因子名: [因子dataframes]}
    #     放成list是为了在最后一步再concat
    #     """
    #     support_factor_agg_dict = {}
    #     for factor in self.name_list:
    #         factor_param_list = self.param_lists[factor]
    #         for param_list in factor_param_list:
    #             factor_name = '_'.join([factor] + [str(x) for x in param_list])
    #             support_factor_agg_dict[factor_name] = [] #  pd.DataFrame(dtype=float)
    #     self.support_factor_agg_dict = support_factor_agg_dict  # 放回self, 方便母进程和子进程读取


    # def unpack_result_dict_into_support_factor_agg_dict(self, result_dict):
    #     """
    #     将子进程回传的result_dict放入support_factor_agg_dict 
    #     """
    #     # 添加因子
    #     print('正在拆包拼接dataframe')
    #     for result_dict_per_ticker in result_dict:
    #         for factor_name, factor_df in result_dict_per_ticker.items():
    #             self.support_factor_agg_dict[factor_name].append(factor_df)
    #     print('完成拆包拼接')
    

    # def save_factor_df_in_support_factor_agg_dict(self):
    #     """
    #     将打包好的support_factor_agg_dict 存入
    #     """
    #     print('开始写入因子')
    #     agg_start = time.time()
    #     for factor_name, factor_df_lst in self.support_factor_agg_dict.items():
    #         start = time.time()
    #         factor_df = concat(factor_df_lst)
    #         factor_df = factor_df + factor_df * 0   # 将inf转换为nan
    #         save_eod_feature(factor_name, factor_df)
    #         print(f'完成因子{factor_name}的写入, 耗时{time.time() - start}')
    #     print(f'完成写入所有因子, 耗时{time.time() - agg_start}')
        
    
    # --------------------- eod ------------------------------------
    # eod 数据计算开了共享内存，有以下几点需要注意：
    # - 不能在shared memory内层print，否则会segmentation fault，建议用其他方式debug
    # - shared memory只支持int和float，不能存str('object')
    # - shared memory 的命名为eod_{字段}_shengy，此乃唯一读取共享内存的密钥。可以在101其他地方读取，但不能两个compute instance同时读取

    def get_eod(self) -> Dict[str, pd.DataFrame]:
        """ get eod data """
        # eod_history = ds.get_eod_history(tickers=self.tickers, 
        #                                  start_date=self.start_date,
        #                                  end_date=self.end_date,
        #                                  fields=self.required_field_types_dict['eod']
        #                                 )
        
        # # 0818: closeprice, adjfactor, 和preclose在数据库里有ffill，生成的时候要用open mask回去
        # # 往前多读一天，保证preclose的第一天不会被多余drop掉
        # open_price = ds.get_eod_history(tickers=self.tickers, 
        #                                 start_date=ds.get_prev_trade_date(trade_date=self.start_date), 
        #                                 end_date=self.end_date, 
        #                                 fields=['OpenPrice']
        #                                )["OpenPrice"]
        # open_price_mask = open_price - open_price
        # open_price_mask_selected = open_price_mask.loc[:, self.start_date:self.end_date]
        # open_price_mask_shifted_selected = open_price_mask.shift(1, axis=1).loc[:, self.start_date:self.end_date]

        # # 加mask
        # eod_history['ClosePrice'] = eod_history['ClosePrice'] + open_price_mask_selected
        # eod_history['AdjFactor'] = eod_history['AdjFactor'] + open_price_mask_selected
        # eod_history['PreClosePrice'] = eod_history['PreClosePrice'] + open_price_mask_selected + open_price_mask_shifted_selected

        # TODO: maybe some preprocessing as above 
        eod_history = ds.get_eod_history(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            fields=self.required_field_types_dict['eod'],
            source='stock'
        )
        return eod_history


    def create_shm(self, to_share, name):
        """
        创建共享内存，将数据存入创建的共享内存
        :param to_share: a dataframe 
        :param name: 读取用的名字
        """
        to_share_np = to_share.to_numpy()
        shm = SharedMemory(create=True, name=name, size=to_share_np.nbytes)
        shm_data = np.ndarray(to_share_np.shape, dtype=to_share_np.dtype, buffer=shm.buf)
        shm_data[:] = to_share_np[:]

    
    def save_eod_to_shms(self, eod_data_dict):
        """ 
        将eod所有字段存入shm，命名为eod_{字段}_{user}，同时也包括eod_column_{user} 和 eod_index_{user}
        :param eod_data_dict: get_eod_history的返回结果
        """
        # 保存不变的量
        self.eod_keys = list(eod_data_dict.keys())
        template = eod_data_dict['ClosePrice']
        template_index = template.index.astype(int)
        template_column = template.columns.astype(int)
        close_price_record = template.to_numpy()
        self.template_shape = close_price_record.shape

        # 存入shm
        self.create_shm(template_index, f'eod_index_{user}')
        self.create_shm(template_column, f'eod_column_{user}')
        for eod_key, df in eod_data_dict.items():
            self.create_shm(df, f'eod_{eod_key}_{user}')
        

    def clean_each_shm(self, eod_key):
        """
        清理单个shm，所有创建的shm都需要close和unlink
        :param eod_key: eod数据字段
        """
        shm = SharedMemory(name=f'eod_{eod_key}_{user}')
        shm.close()
        shm.unlink()
    

    def clean_all_shm(self):
        """ 
        清理所有shm
        """
        # 清理shm
        self.clean_each_shm('index')
        self.clean_each_shm('column')
        for key in self.eod_keys:
            self.clean_each_shm(key)        


    def process_each_eod_func(self, factor, param_list):
        """
        子进程: 
            - 重构stock_data_dict
            - 计算因子值
            - 返回因子值dataframe
        :param factor_name: the name of the factor (also a function in either factor_gen or support_factor_gen) 
        :param param_list: the param_list to be forwarded to the function 
        :return the factor dataframe
        """
        # 读取index和column
        start = time.time()
        saved_index_shm = SharedMemory(name=f'eod_index_{user}')
        saved_index_np = np.ndarray((self.template_shape[0], ), dtype='int', buffer=saved_index_shm.buf)
        saved_column_shm = SharedMemory(name=f'eod_column_{user}')
        saved_column_np = np.ndarray((self.template_shape[1], ), dtype='int', buffer=saved_column_shm.buf)

        # 转换index和column为string
        index = [str(x).zfill(6) for x in saved_index_np]
        column = [str(x) for x in saved_column_np]

        # 重构eod_data_dict
        reassembled_eod_data_dict = {} 
        for eod_key in self.eod_keys:
            saved_shm = SharedMemory(name=f'eod_{eod_key}_{user}')
            saved_np = np.ndarray(self.template_shape, dtype='float64', buffer=saved_shm.buf).copy()
            saved_df = pd.DataFrame(saved_np, index=index, columns=column)
            reassembled_eod_data_dict[eod_key] = saved_df
        
        factor_name = '_'.join([factor] + [str(x) for x in param_list])
        
        # 拼装stock_data_dict
        stock_data_dict = {'eod': reassembled_eod_data_dict}
        
        # 计算因子
        factor_df = eval(f'{self.source}.{factor}(stock_data_dict, param_list)')
        factor_df = factor_df + factor_df * 0
        ds.save_eod_feature(factor_name, factor_df, des=self.des)
        print(f'计算和储存因子{factor_name}', time.time() - start)


    def run_eod(self):
        """ 运行eod类因子生成 """
        # 读取eod数据
        start = time.time()
        eod_data_dict = self.get_eod()
        print('读取eod数据耗时', time.time() - start)

        # 存入shared memory
        self.save_eod_to_shms(eod_data_dict)
        print('SHM Ready')

        # 等待共享内存完全建立（即便已经print了shm ready，共享内存并未完全存入，需手动等待）
        time.sleep(1)

        # 生成多进程
        process_list = []
        factor_names = []
        failed_messages = []
        failed_factor_list = []
        start = time.time()
        pool = mp.Pool(processes=8)
        for factor in self.name_list:
            factor_param_list = self.param_lists[factor]
            for param_list in factor_param_list:
                process_list.append(pool.apply_async(self.process_each_eod_func, args=(factor, param_list)))
                factor_names.append('_'.join([factor] + [str(x) for x in param_list]))

        # 等待进程完全传入
        time.sleep(1)
        # 运行多进程
        for i in range(len(process_list)):
            try:
                process_list[i].get()
            except:
                failed_messages.append(traceback.format_exc())
                failed_factor_list.append(factor_names[i])
                
        pool.close()
        pool.join()
        print('完成多进程', time.time() - start)

        # log加入未成功的factors
        if failed_factor_list: 
            curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('src/factor_generator/log/failed_factors.txt', 'a') as f:
                f.write(curr_time + '\n')
                for failed_factor, failed_message in zip(failed_factor_list, failed_messages):
                    f.write(failed_factor + '\n')
                    f.write(failed_message + '\n')
                    print(failed_message)
                f.write('\n')
        
        # 清理shm
        start = time.time()
        self.clean_all_shm()
        print(f'完成共享内存清理, 耗时{time.time() - start}')
    
    # ---------------------------------------------
    # -------------------- 总体运行 ----------------
    # ---------------------------------------------

    def run(self):
        """ run factor generation """
        # EOD
        if 'eod' in self.required_data_types:
            try: 
                self.run_eod()
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print('eod共享内存失败, 清除shm')
                self.clean_all_shm()
        
        # 运行财务基本面数据
        elif 'fund' in self.required_data_types:
            self.run_fund()
        
        # 如不需要读取任何数据
        else:
            self.run_no_required_data()
