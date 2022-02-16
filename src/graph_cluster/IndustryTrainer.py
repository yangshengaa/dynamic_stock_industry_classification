"""
rolling-based dynamic industry training. Train test periods coincide with the machine learning part 

the procedure is as follows: 
1. select stock pools (take stocks ever appeared in both train and test periods 
    (e.g. zz1000 would have roughly 1200 stocks in the 240 + 40 days period))
2. build graph on the 240 days; 
3. detect communities and take that as the industry classification for the 40 days. 
4. concat industry classification 

Note that 2 is done in GraphCollection.py, and 3 is done in CommunityDetectionUtils.py
"""

# load packages 
import os 
import time
import pickle 
import logging
from typing import List, Dict
import numpy as np
import pandas as pd

# load files 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
import src.graph_cluster.config as cfg 

# specify paths 
TRAIN_TEST_DATES_PATH = 'out/train_test_dates'

class IndustryTrainer:

    def __init__(
        self, 
        graph_type: str = None, 
        num_clusters: int = None,
        clustering_type: str = None, 
        filter_mode: int = None
    ) -> None:
        # init dataserver 
        self.ds = PqiDataSdkOffline()
        # load config 
        if graph_type is None: 
            self.graph_type = cfg.graph_type
        else: 
            self.graph_type = graph_type 
        
        if filter_mode is None: 
            self.filter_mode = cfg.filter_mode
        else: 
            self.filter_mode = filter_mode 
        
        if clustering_type is None:
            self.clustering_type = cfg.clustering_type
        else: 
            self.clustering_type = clustering_type 
        
        if num_clusters is None: 
            self.num_clusters = cfg.num_clusters
        else: 
            self.num_clusters = num_clusters
        
        self.stock_pool = cfg.stock_pool

        # read index mask 
        self.index_mask = self.ds.get_index_mask([self.stock_pool])

        # select graph model 
        self.select_graph_model()

        # figure out name to save 
        self.determine_save_name()
    
    def determine_save_name(self):
        """ 
        figure out the name to save the ind feature 
        {stock_pool}_{num_clusters}_{graph_type}_{filter_mode}_{clustering_type}_{*params}
        """
        # base 
        name = f'{self.stock_pool}_{self.num_clusters}_{self.graph_type}_{self.filter_mode}_{self.clustering_type}_'
        # add vector embedding params
        if self.clustering_type == 'node2vec':
            name += '_'.join([str(x) for x in cfg.node2vec_rw_params.values()] + [str(x) for x in cfg.node2vec_word2vec_params.values()])
        elif self.clustering_type == 'sub2vec':
            name += '_'.join(
                [str(cfg.sub2vec_walk_length), str(cfg.num_hops), str(cfg.sub2vec_mode)] 
              + [str(x) for x in cfg.sub2vec_params.values()]
            )
        
        self.name = name

    def select_graph_model(self):
        """ select graph model according to config """

        if self.graph_type == 'MST':
            from src.graph_cluster.GraphCollections import MST
            graph_model = MST 

        elif self.graph_type == 'AG':
            from src.graph_cluster.GraphCollections import AG
            graph_model = AG 

        elif self.graph_type == 'PMFG':
            from src.graph_cluster.GraphCollections import PMFG 
            graph_model = PMFG 

        else: 
            raise NotImplementedError(f'Graph Type {self.graph_type} not defined')
        
        self.graph_model = graph_model

    def train_industry_each_period(
            self, 
            train_dates: List[str], test_dates: List[str]
        ) -> pd.DataFrame:
        """ 
        in each train period: 
        - load data
        - feed into graphs 
        - output clustering

        :param train_dates, test_dates: array of dates
        :return the clustering on the test periods 
        """
        print(f'======== Training {train_dates[0]} - {train_dates[-1]}  ========')
        start = time.time()
        # read train data 
        eod_data_dict = self.ds.get_eod_history(
            start_date=train_dates[0], end_date=train_dates[-1], 
            fields=['ClosePrice', 'AdjFactor']
        )
        close_df = eod_data_dict['ClosePrice'] * eod_data_dict['AdjFactor']
        return_df = close_df / close_df.shift(1, axis=1) - 1

        # prune return df (subset to selected stock pool, e.g. zz1000)
        # index_mask_selected = self.index_mask[train_dates]
        # * select all stocks ever in the pool, including test. Note that this is not future-gazing, 
        # * since whenever a new stock is added to the pool, a retrain is necessary. 
        index_mask_selected = self.index_mask[list(train_dates) + list(test_dates)]
        member_stock_mask = index_mask_selected.loc[index_mask_selected.notna().any(axis=1)]
        member_stock_list = member_stock_mask.index.tolist()
        nonmember_stock_list = list(set(close_df.index) - set(member_stock_list))
        member_stock_return_df = return_df.loc[member_stock_list]

        # count number of stocks in test dates but not in train pool 
        # test_index_mask_selected = self.index_mask[test_dates]
        # test_member_stock_mask = test_index_mask_selected.loc[test_index_mask_selected.notna().any(axis=1)]
        # test_member_stock_list = test_member_stock_mask.index.tolist()
        # not_appeared_stocks = list(set(test_member_stock_list) - set(member_stock_list))
        # print(f'number of stocks not appeared in train: {len(not_appeared_stocks)}')
        # print(f'Stocks in test not appeared in train: {not_appeared_stocks}')

        print(f'number of member/non-member stocks: {len(member_stock_list)}, {len(nonmember_stock_list)}')

        # feed into tree model 
        graph = self.graph_model(
            member_stock_return_df, 
            num_clusters=self.num_clusters,
            clustering_type=self.clustering_type,
            filter_mode=self.filter_mode
        )
        label_dict = graph.detect_community()

        print(f'write to {test_dates[0]} - {test_dates[-1]}, taking {time.time() - start:.3f}s\n')

        # put back into a cross-sectional series 
        member_stock_clustering = pd.Series(label_dict.values(), index=label_dict.keys())
        nonmember_stock_clustering = pd.Series(-1, index=nonmember_stock_list)
        cross_sectional_clustering = pd.concat([member_stock_clustering, nonmember_stock_clustering])
        cross_sectional_clustering.sort_index(inplace=True)  # sort tickers 

        # concat to dataframe
        test_period_clustering_arr = np.vstack([cross_sectional_clustering.tolist()] * len(test_dates))
        test_period_clustering_df = pd.DataFrame(
            test_period_clustering_arr.T, 
            index=cross_sectional_clustering.index, 
            columns=test_dates
        )

        return test_period_clustering_df


    def train_industry(self, train_test_dates):
        """ 
        aggregate train and test 
        
        :param train_test_dates: List(Tuple(train_dates, test_dates))
            note that test dates are consecutive 
        :return concat dataframe 
        """
        # train test each period 
        ind_df_list = [] 
        for train_dates, test_dates in train_test_dates:
            ind_df_each_period = self.train_industry_each_period(train_dates, test_dates)
            ind_df_list.append(ind_df_each_period)
        
        # concat 
        ind_df = pd.concat(ind_df_list, axis=1)

        # bfill for the first period (meaningless but for sanity)
        pre_dates = list(set(self.ds.trade_dates) - set(ind_df.columns))
        pre_dates_df = pd.DataFrame(np.nan, index=ind_df.index, columns=pre_dates)
        complete_ind_df = pd.concat([pre_dates_df, ind_df], axis=1)
        complete_ind_df = complete_ind_df.bfill(axis=1)
        return complete_ind_df

    # --------------- io ---------------
    def load_train_test_dates(self):
        """ read train test dates """
        with open(os.path.join(TRAIN_TEST_DATES_PATH, 'train_test_dates.pkl'), 'rb') as f:
            train_test_dates = pickle.load(f)
        return train_test_dates

    def save_trained_industry(self, ind_df: pd.DataFrame):
        """ write to disk """
        self.ds.save_ind_feature(self.name, ind_df)
        print(f'Save to {self.name}')

    def run(self):
        """ main function to run """
        start = time.time()
        train_test_dates = self.load_train_test_dates()
        ind_df = self.train_industry(train_test_dates)
        self.save_trained_industry(ind_df)
        print(f'Clustering takes {time.time() - start:.3f}s in total')
