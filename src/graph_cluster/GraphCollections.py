"""
Graph Collections: 
- Asset Graph (AG)
- Minimum Spanning Tree (MST)
- Planar Maximally Filtered Graph (PMFG)
- Random Matrix Theory (RMT) 

RMT in particular would not generate a graph but a community directly.

Many algorithms are discussed in Community Detection for Correlation Matrices by Mel MacMahon and Diego Garlaschelli.
"""

# load packages 
import os 
import traceback
import logging
from typing import Dict, List

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms.community import girvan_newman


from sklearn.cluster import AgglomerativeClustering

# load files 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
import src.graph_cluster.config as cfg
from src.graph_cluster.similarity_measures import *
from src.graph_cluster.CommunityDetectionUtils import (
    Node2Vec,
    Sub2Vec
)

# set chinese font 
CN_FONT = 'Heiti TC'


# =================================== 
# --------- abstract class ----------
# ===================================

class GeneralGraph:

    def __init__(self, return_df: pd.DataFrame, num_clusters: int=10) -> None:
        # load data 
        self.return_df = return_df
        self.num_clusters = num_clusters
        self.is_graph = True  # True for AG, MST, PMFG, False for RMT 

        # compute similarity 
        self.similarity_metric = cfg.similarity_metric
        self.compute_similarity()
    
    def compute_similarity(self):
        """ compute similarity from return_df """
        similarity_df = eval(f'{self.similarity_metric}(self.return_df)')
        self.similarity_df = similarity_df        
    
    def build_graph(self):
        raise NotImplementedError()
    
    def detect_community(self, g:nx.Graph=None) -> Dict[str, int]:
        raise NotImplementedError()

    def visualize(self):
        raise NotImplementedError()
    

# ===================================
# --------- sub classes -------------
# ===================================
class AG(GeneralGraph):

    def __init__(self, return_df: pd.DataFrame, num_clusters: int = 10) -> None:
        super().__init__(return_df, num_clusters)
        self.clustering_type = cfg.clustering_type
        assert self.similarity_metric == 'cor', 'Asset Graph Only Accepts Correlation Matrix'  
    
    def build_graph(self) -> nx.Graph: 
        """ build asset graph """
        # set up parameters 
        tau = 10   # 2 in the original paper, but for T = 240, this may be better
        T = self.return_df.shape[1]
        C_tau = (np.exp(2 * tau / np.sqrt(T - 3)) - 1) / (np.exp(2 * tau / np.sqrt(T - 3)) + 1)  # threshold 

        # empty graph 
        g = nx.Graph()
        tickers = self.similarity_df.index.tolist()
        for ticker in tickers: 
            g.add_node(ticker)
        
        # filter correlation info 
        upper_idx = np.triu_indices(self.similarity_df.shape[0], k=1)
        upper_idx_tuple = np.array(list(zip(*upper_idx)))
        similarities_np = self.similarity_df.values[upper_idx]
        for (upper_idx_i, upper_idx_j), cor_ij in zip(upper_idx_tuple, similarities_np):
            if abs(cor_ij) > C_tau:
                g.add_edge(tickers[upper_idx_i], tickers[upper_idx_j])
        
        return g

    def detect_community(self, g: nx.Graph=None) -> Dict[str, int]:
        """ Naive: Node2Vec + KMeans """
        # get graph 
        if g is None:
            g = self.build_graph()
        
        # clustering model
        if self.clustering_type == 'sub2vec':
            clustering_model = Sub2Vec
        elif self.clustering_type == 'node2vec':
            clustering_model = Node2Vec
        else:
            raise NotImplementedError('Not Yet Implemented')
        model = clustering_model(g, num_clusters=self.num_clusters)
        model.generate_embeddings()
        label_dict = model.get_community()
        
        return label_dict
                
    def visualize(
        self,
        g: nx.Graph,
        label_dict: Dict[str, int],
        custom_name: str = '',
        use_cn_name: bool=False
    ):
        """ visualize asset graph with community labels """
        # select subsets
        nodelist = list(label_dict.keys())
        node_color = [label_dict[node] for node in nodelist]

        # draw 
        fig, ax = plt.subplots(figsize=(16, 12))
        drawing_params = {  # nodes
            "node_size" : 150,
            "nodelist" : list(label_dict.keys()),
            "node_color" : node_color,

            # edges
            "width" : 0.8,
            "edge_color" : "gainsboro",

            # labels
            "with_labels" : True,
            "font_size" : 3,
            "cmap" : 'Accent'
        }
        if use_cn_name:
            drawing_params['labels'] = PqiDataSdkOffline.get_ticker_name_cn()
            drawing_params['font_family'] = CN_FONT

        nx.draw_spring(g, **drawing_params)
        ax.set_title('Asset Graph for Stocks {}'.format(custom_name), fontsize=20)
        fig.tight_layout()
        plt.savefig(os.path.join(cfg.fig_save_path, f'ag_{custom_name}.png'), dpi=800)

class MST(GeneralGraph):

    def __init__(self, return_df: pd.DataFrame, num_clusters: int = 10) -> None:
        super().__init__(return_df, num_clusters=num_clusters)
    
    def build_graph(self) -> nx.Graph:
        """ build MST (Kruskal's Algorithm) """
        # empty graph 
        g = nx.Graph()
        tickers = self.similarity_df.index.tolist()
        num_tickers = len(tickers)
        for ticker in tickers:
            g.add_node(ticker)

        # sort similariies in descending order and get stock code pairs
        upper_idx = np.triu_indices(self.similarity_df.shape[0], k=1)
        upper_idx_tuple = np.array(list(zip(*upper_idx)))
        similarities_np = self.similarity_df.values[upper_idx]
        similarities_np[np.isnan(similarities_np)] = -999  # let nan be the least ones
        sorted_pair_idx = upper_idx_tuple[np.argsort(similarities_np)[::-1]]
        sorted_pair_stock_code = [(tickers[i1], tickers[i2]) for i1, i2 in sorted_pair_idx]

        # special handling: if too many nans, do not connect them to the main tree
        na_stock_counts = (self.similarity_df.isna().mean() > cfg.na_threshold).sum()

        # build graph 
        num_edges = 0
        for stock_code_1, stock_code_2 in sorted_pair_stock_code:
            # if two nodes are not connected, put an edge in between
            if not nx.has_path(g, stock_code_1, stock_code_2):
                g.add_edge(stock_code_1, stock_code_2)
                num_edges += 1

                # early stopping: N - 1 edges, then MST is ready 
                if num_edges == num_tickers - 1 - na_stock_counts:  
                    break        
        return g

    def detect_community(self, g: nx.Graph=None) -> Dict[str, int]:
        """ 
        Complete Linkage Algorithm. Note that this does not depend on built graph 
        note that single linkage seems not working for AgglomerativeClustering in precomputed mode
        
        :return dict[stock_code, community label]
        """
        # compute distance frm similarity 
        logging.warning('distance measures in mst only take correlation')
        distance_df = np.sqrt(2 * (1 - self.similarity_df.fillna(0)))   # assumption is that nan means uncorrelated 
        distance_df = distance_df - np.diag(np.diag(distance_df))  # if fillna on diag, make sure to turn it back to 0

        # clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.num_clusters,
            affinity='precomputed',
            linkage='complete'
        ).fit(distance_df)

        # output labels
        community_labels = clustering.labels_
        label_dict = dict(zip(self.similarity_df.index, community_labels))
        return label_dict

    def visualize(
        self, 
        g: nx.Graph, 
        label_dict: Dict[str, int], 
        custom_name: str='',
        use_cn_name: bool=False
    ):
        """ visualize MST: advised to plot only a subset of data (e.g. zz1000) """

        # select subsets 
        nodelist = list(label_dict.keys())
        node_color = [label_dict[node] for node in nodelist]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        drawing_params = {
            # nodes
            'node_size': 150,
            'nodelist': list(label_dict.keys()),
            'node_color': node_color,

            # edges
            'width': 0.8,
            'edge_color': "gainsboro",

            # labels
            'with_labels': True,
            'font_size': 3,
            'cmap': 'Accent'
        }
        if use_cn_name:
            drawing_params['labels'] = PqiDataSdkOffline.get_ticker_name_cn()
            drawing_params['font_family'] = CN_FONT
        
        nx.draw_spring(g, **drawing_params)

        ax.set_title('MST for Stocks {}'.format(custom_name), fontsize=20)
        fig.tight_layout()
        plt.savefig(os.path.join(cfg.fig_save_path, f'mst_{custom_name}.png'), dpi=800)
