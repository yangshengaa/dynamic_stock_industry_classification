"""
Same testing procedure as batch_factor test 
"""

# load packages 
import os
import sys
import time
import traceback
import logging

import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

# load files 
from src.backtest.configuration import config as cfg
from src.backtest.tools.datatools import DataAssist
from src.backtest.bin.single_signal_test import SingleSignalBacktest

# init data
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline


logging.basicConfig(level=logging.CRITICAL)


class BasicBatchTest:
    def __init__(self):
        # self.myconnector = PqiDataSdk(user=cfg.user, size=20, pool_type="mp", log=False, offline=True)
        self.myconnector = PqiDataSdkOffline()
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        self.stock_pool = self.myconnector.get_ticker_list()  # read all and mask later
        self.usr = cfg.user
        self.index_list = cfg.index_list
        pool_type = " + ".join(self.index_list) + ', ' + ('fmv weighted' if cfg.weight_index_by_fmv else 'equally weighted')
        self.pool_type = pool_type
        self.fix_stocks = cfg.fix_stocks
        self.fixed_stock_pool = cfg.fixed_stock_pool
        print(f'Stock Pool: {self.pool_type}')
        self.max_processes = cfg.max_processes
        self.shape = []
        self.ind_shape = []
        self.factor_dict = {}
        self.key_list = []
        self.all_key_list = []
        self.namespace = ""
        self.name_list = []
        self.factor_path = cfg.factor_path
        self.read_from_feather = cfg.read_from_feather


    def run(self, namespace="", name_list=None):
        self.namespace = namespace
        self.name_list = name_list
        self.load_data()
        self.read_factor()
        print("SHM ready")

    def load_data(self):
        # read data
        tester = SingleSignalBacktest()
        eod_data_dict = tester.get_data()
        # save ndarray to shm
        df_sample = eod_data_dict["ClosePrice"]
        self.df_sample = df_sample
        self.date_list = list(df_sample.columns)
        self.shape = df_sample.shape
        self.ind_shape = eod_data_dict["ind_df"].shape
        self.index_shape = eod_data_dict["index_data"].shape
        self.calendar_shape = eod_data_dict["calendar"].shape
        index_array = np.array(df_sample.index.T).astype("int")
        columns_array = np.array(df_sample.columns.T).astype("int")
        index_array_ind = np.array(eod_data_dict["ind_df"].index.T).astype("int")
        columns_array_ind = np.array(eod_data_dict["ind_df"].columns.T).astype("int")        
        index_index_data_array = np.array(eod_data_dict["index_data"].index.T).astype("int")
        columns_index_data_array = np.array(eod_data_dict["index_data"].columns.T).astype("int")
        calendar = np.array(eod_data_dict["calendar"].astype("int"))
        abnormal_keys = []
        for k in eod_data_dict.keys():
            if eod_data_dict[k].shape != self.shape and k != "ind_df" and k != 'calendar' and k != 'index_data':
                print("delete {}  with shape {}".format(k, eod_data_dict[k].shape))
                abnormal_keys.append(k)

        for k in abnormal_keys:
            del eod_data_dict[k]

        self.key_list = list(eod_data_dict.keys())
        self.all_key_list = self.key_list.copy()
        self.all_key_list.extend(["index", "columns", "index_ind", "columns_ind"])
        self.save_to_shm(index_array, "index"),
        self.save_to_shm(columns_array, "columns")
        self.save_to_shm(index_array_ind, "index_ind")
        self.save_to_shm(columns_array_ind, "columns_ind")
        self.save_to_shm(calendar,"calendar")

        self.all_key_list.extend(["index_index_data", "columns_index_data"])
        self.save_to_shm(index_index_data_array, "index_index_data")
        self.save_to_shm(columns_index_data_array, "columns_index_data")

        for k in self.key_list:
            if k != 'calendar':
                data = eod_data_dict[k].values
                self.save_to_shm(data, k)
    
    def read_factor(self):
        """
        load factors
        """
        self.factor_dict = self.read_factor_data(self.name_list, self.stock_pool, self.date_list)


    def read_factor_data(self, test_factor_list, tickers, date_list):
        """
        read index 
        """
        # read index mask 
        index_mask = DataAssist.get_index_mask(self.index_list)

        # if read from feather
        if self.read_from_feather:
            path = self.factor_path
            factor_dict = {}
            for feather_path in test_factor_list:
                # raw_factor_df = pd.read_csv(path + csv_path, index_col=0)
                raw_factor_df = pd.read_feather(os.path.join(path, feather_path)).set_index('index')
                # raw_factor_df.index = [str(x).zfill(6) for x in raw_factor_df.index]
                raw_factor_df = raw_factor_df + (self.df_sample - self.df_sample) # algin index and dates
                raw_factor_df = raw_factor_df[self.df_sample.columns]

                # for weight and equal signal，scale to sum=1 
                raw_factor_df = raw_factor_df / raw_factor_df.sum()
                raw_factor_df = raw_factor_df + raw_factor_df * 0   # remove inf
                
                # dynamic pool 
                # TODO: better signal name
                factor_name = feather_path.replace('/', '_')
                if not self.fix_stocks:
                    factor_dict[factor_name] = raw_factor_df * index_mask 
                # fixed_pool
                else:
                    factor_dict[factor_name] = raw_factor_df + (raw_factor_df.loc[self.fixed_stock_pool] - raw_factor_df.loc[self.fixed_stock_pool])

        else:
            path = self.factor_path
            # feature_name_list = ["eod_" + feature_name for feature_name in test_factor_list]
            factor_dict = {}
            # factors = self.myconnector.get_eod_feature(fields=feature_name_list,
            #                                         where=path,
            #                                         tickers=tickers,
            #                                         dates=date_list)

            # add mask
            for factor in test_factor_list:
                # raw_factor_df = factors[factor].to_dataframe()
                raw_factor_df = self.myconnector.read_eod_feature(
                    factor, des=factor, dates=date_list
                )

                # for weight and equal signal，scale to sum=1 
                raw_factor_df = raw_factor_df / raw_factor_df.sum()
                raw_factor_df = raw_factor_df + raw_factor_df * 0   # remove inf
                
                # dynamic pool
                if not self.fix_stocks:
                    factor_dict[factor] = raw_factor_df * index_mask
                # fixed pool
                else:
                    factor_dict[factor] = raw_factor_df + (raw_factor_df.loc[self.fixed_stock_pool] - raw_factor_df.loc[self.fixed_stock_pool])
        return factor_dict



    def save_to_shm(self, data_nd, name):
        """
        read ndarray and save to shm 

        :param data_nd:
        :param name:
        :return:
        """
        name = name + "," + self.usr
        shm_address = SharedMemory(name=name, create=True, size=data_nd.nbytes)
        shm_nd_data = np.ndarray(data_nd.shape, dtype=data_nd.dtype, buffer=shm_address.buf)
        shm_nd_data[:] = data_nd[:]

    def shmClean(self):
        # earse shm 
        for k in self.all_key_list:
            try:
                shm = SharedMemory(name=k + "," + self.usr)
                shm.close()
                shm.unlink()
            except Exception as e:
                print(e)

# do not append to the class as a class method. 
def processor(factor, shape, ind_shape, index_shape, calendar_shape, usr, key_list):
    """
    single process
    :param factor:
    :param from_ds:
    :param key_list:
    :param usr:
    :param ind_shape:
    :param shape:
    :return:
    """
# reconstruct data_dict
    data_dict = dict()
    try:
        shm_index = SharedMemory(name='index' + "," + usr)
        index = np.ndarray((shape[0],), dtype='int', buffer=shm_index.buf)
        index = [str(x).zfill(6) for x in index]
        shm_columns = SharedMemory(name='columns' + "," + usr)
        columns = np.ndarray((shape[1],), dtype='int', buffer=shm_columns.buf)
        columns = [str(x) for x in columns]

        shm_index_ind = SharedMemory(name='index_ind' + "," + usr)
        index_ind = np.ndarray((ind_shape[0],), dtype='int', buffer=shm_index_ind.buf)
        index_ind = [str(x).zfill(6) for x in index_ind]
        shm_columns_ind = SharedMemory(name='columns_ind' + "," + usr)
        columns_ind = np.ndarray((ind_shape[1],), dtype='int', buffer=shm_columns_ind.buf)
        columns_ind = [str(x).zfill(6) for x in columns_ind]

        shm_index_index_data = SharedMemory(name='index_index_data' + "," + usr)
        index_index_data = np.ndarray((index_shape[0],), dtype='int', buffer=shm_index_index_data.buf)
        index_index_data = [str(x) for x in index_index_data]
        shm_columns_index_data = SharedMemory(name='columns_index_data' + "," + usr)
        columns_index_data = np.ndarray((index_shape[1],), dtype='int', buffer=shm_columns_index_data.buf)
        columns_index_data = [str(x) for x in columns_index_data]

        for key in key_list:
            key = key + "," + usr
            # industry dataframe
            if key == "ind_df" + "," + usr:
                shm_temp = SharedMemory(name=key)
                data_temp = np.ndarray(ind_shape, dtype='float64', buffer=shm_temp.buf).copy()
                key = key.split(",")[0]
                data_dict[key] = pd.DataFrame(data=data_temp, index=index_ind, columns=columns_ind)

            # index return ataframe
            elif key == "index_data" + "," + usr:
                shm_temp = SharedMemory(name=key)
                data_temp = np.ndarray(index_shape, dtype='float64', buffer=shm_temp.buf).copy()
                key = key.split(",")[0]
                data_dict[key] = pd.DataFrame(data=data_temp, index=index_index_data, columns=columns_index_data)
            
            # calendar ataframe
            elif key == "calendar" + "," + usr:
                shm_temp = SharedMemory(name=key)
                data_temp = np.ndarray(calendar_shape, dtype='float64', buffer=shm_temp.buf).copy()
                key = key.split(",")[0]
                data_dict[key] = data_temp

            # eod_data_dict
            else:
                shm_temp = SharedMemory(name=key)
                data_temp = np.ndarray(shape, dtype='float64', buffer=shm_temp.buf).copy()
                key = key.split(",")[0]
                data_dict[key] = pd.DataFrame(data=data_temp, index=index, columns=columns)

    except Exception as e:
        error_message = "Signal {} shm failed: {}".format(factor[1], e) 
        print(error_message)
        error_message_complete = error_message + '\n' +  traceback.format_exc()
        return error_message_complete

    # signal backtest
    try:
        backtester = SingleSignalBacktest(offline=True) # daemonic processes are not allowed to have children
        backtester.get_data_multi(data_dict)
        return (backtester.run_signal(factor[0], factor[1]))

    except Exception as e:
        error_message = "Signal {} backtest failed: {}".format(factor[1], e)
        print(error_message)
        error_message_complete = error_message + '\n' +  traceback.format_exc()
        return error_message_complete

def save_record(summary_df):
    d = cfg.output_path
    curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
    # create a new folder 
    save_path = '{}/signal_test_file_{}_{}'.format(d, curr_date, cfg.test_name)
    summary_file = '{}/summary.csv'.format(save_path)
    try:
        final_summary_df = pd.read_csv(summary_file)
        del final_summary_df['Unnamed: 0']
        final_summary_df = pd.concat([final_summary_df,summary_df])
        final_summary_df.to_csv(summary_file)
    except FileNotFoundError:
        summary_df.to_csv(summary_file)

def run():
    """
    main function
    """
    signal_name_list = cfg.signal_name_list
    res_list = []
    batch_tester = BasicBatchTest()
    try: 
        batch_tester.run(namespace="", name_list=signal_name_list)

        # define processes
        pool = mp.Pool(processes=cfg.max_processes)
        for i in range(len(batch_tester.factor_dict.keys())):
            factor_name = list(batch_tester.factor_dict.keys())[i]
            factor_df = batch_tester.factor_dict[factor_name]
            res_list.append(pool.apply_async(processor, args=(
                [factor_df, factor_name], 
                batch_tester.shape,
                batch_tester.ind_shape, 
                batch_tester.index_shape, 
                batch_tester.calendar_shape, 
                batch_tester.usr, 
                batch_tester.key_list,
            )))

        time.sleep(1)

        res_group_list = []
        for res in res_list:
            res_group_list.append(res.get())

        pool.close()
        pool.join()

        final_res_group_list = []
        for res in res_group_list:
            if res:
                # if failed, print 
                if 'Traceback' in res:
                    print(res)
                # add to final reports
                elif len(res) != 0:
                    final_res_group_list.append(res)
        res_group_list = final_res_group_list

        # record factors
        summary_df = pd.DataFrame(np.array(res_group_list), columns=cfg.signal_summary_cols)
        save_record(summary_df)
    except Exception as e: 
        print(e)
        print(traceback.format_exc())
    finally:
        # clean shm
        batch_tester.shmClean()
