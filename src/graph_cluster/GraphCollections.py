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
import logging
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import networkx as nx

import planarity

from sklearn.cluster import AgglomerativeClustering, SpectralClustering

# load files 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
import src.graph_cluster.config as cfg
from src.graph_cluster.similarity_measures import *
from src.graph_cluster.CommunityDetectionUtils import Node2Vec, Sub2Vec, ModifiedLouvain


# logging config 
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', 
    # level=logging.INFO
)

# set chinese font 
CN_FONT = 'Heiti TC'
EPS = 1e-15


# =================================== 
# --------- abstract class ----------
# ===================================

class GeneralGraph:

    def __init__(
            self, 
            return_df: pd.DataFrame,
            num_clusters: int=10,
            clustering_type: str=None,
            filter_mode: int=None,
        ) -> None:
        # load data 
        self.return_df = return_df
        self.num_clusters = num_clusters
        self.is_graph = True  # True for AG, MST, PMFG, False for RMT 
        self.clustering_type = cfg.clustering_type if clustering_type is None else clustering_type

        # compute similarity 
        self.similarity_metric = cfg.similarity_metric
        self.filter_mode = cfg.filter_mode if filter_mode is None else filter_mode
        self.compute_similarity()
    
    def compute_similarity(self):
        """ compute similarity from return_df """
        # compute raw similarity 
        similarity_df = eval(f'{self.similarity_metric}(self.return_df)')
        # filter information 
        filtered_similarity_df = ModifiedLouvain.filter_information(
            similarity_df, self.filter_mode, T=self.return_df.shape[0]
        )

        self.similarity_df = filtered_similarity_df        
    
    def build_graph(self) -> nx.Graph:
        raise NotImplementedError()
    
    def detect_community(self, g:nx.Graph=None) -> Dict[str, int]:
        """ detect communities """
        # pick type and run 

        # clustering not using graph structures (but is related to it)
        if self.clustering_type == 'single_linkage':
            # single linkage does not depend on graph, so no graph needed
            # compute distance frm similarity
            logging.warning('distance measures in single linkage only take correlation')
            # enforce correlation after filtering
            cutoff_similarity_df = self.similarity_df.clip(lower=-1 + EPS, upper=1 - EPS) 
            # assumption is that nan means uncorrelated
            distance_df = np.sqrt(2 * (1 - cutoff_similarity_df.fillna(0)))
            # if fillna on diag, make sure to turn it back to 0
            distance_df = distance_df - np.diag(np.diag(distance_df))

            # clustering
            clustering = AgglomerativeClustering(
                n_clusters=self.num_clusters,
                affinity='precomputed',
                linkage='complete'
            ).fit(distance_df)

            # output labels
            community_labels = clustering.labels_
            label_dict = dict(zip(self.similarity_df.index, community_labels))

        # clustering using graph structure 
        else: 
            if g is None:
                g = self.build_graph()
            
            if self.clustering_type == 'spectral':
                clustering = SpectralClustering(
                    n_clusters=self.num_clusters,
                    affinity='precomputed',
                    assign_labels='discretize'
                ).fit(nx.to_numpy_array(g))

                # output labels 
                community_labels = clustering.labels_
                label_dict = dict(zip(self.similarity_df.index, community_labels))
            
            elif self.clustering_type == 'node2vec':
                model = Node2Vec(g, num_clusters=self.num_clusters)
                model.generate_embeddings()
                label_dict = model.get_community()

            elif self.clustering_type == 'sub2vec':
                model = Sub2Vec(g, num_clusters=self.num_clusters)
                model.generate_embeddings()
                label_dict = model.get_community()

            else: 
                raise NotImplementedError(f'{self.clustering_type} not supported')
        
        return label_dict

    def visualize(
        self,
        g: nx.Graph,
        label_dict: Dict[str, int],
        custom_name: str = '',
        use_cn_name: bool = False
    ):
        """ visualize graph with community labels """
        # select subsets
        nodelist = list(label_dict.keys())
        node_color = [label_dict[node] for node in nodelist]

        # draw
        fig, ax = plt.subplots(figsize=(16, 12))
        drawing_params = {  # nodes
            "node_size": 150,
            "nodelist": list(label_dict.keys()),
            "node_color": node_color,

            # edges
            "width": 0.8,
            "edge_color": "gainsboro",

            # labels
            "with_labels": True,
            "font_size": 3,
            "cmap": 'Accent'
        }
        if use_cn_name:
            # extract cn names 
            all_labels = PqiDataSdkOffline.get_ticker_name_cn()
            selected_labels = {}
            for ticker, label in all_labels.items():
                if ticker in nodelist:
                    selected_labels[ticker] = label
            drawing_params['labels'] = selected_labels
            drawing_params['font_family'] = CN_FONT

        nx.draw_spring(g, **drawing_params)
        ax.set_title(
            f'{self.__class__.__name__} {self.clustering_type} filter mode {self.filter_mode} for Stocks {custom_name}', 
            fontsize=20
        )
        fig.tight_layout()
        plt.savefig(
            os.path.join(
                cfg.fig_save_path, f'{self.__class__.__name__}_{self.clustering_type}_{self.filter_mode}_{custom_name}.png'),
            dpi=800
        )
    

# ===================================
# --------- sub classes -------------
# ===================================
class AG(GeneralGraph):

    def __init__(self,
                 return_df: pd.DataFrame,
                 num_clusters: int = 10,
                 clustering_type: str = None,
                 filter_mode: int = None
                 ) -> None:
        super().__init__(return_df, num_clusters, clustering_type=clustering_type, filter_mode=filter_mode)
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

class MST(GeneralGraph):

    def __init__(self,
                 return_df: pd.DataFrame,
                 num_clusters: int = 10,
                 clustering_type: str = None,
                 filter_mode: int = None
                 ) -> None:
        super().__init__(return_df, num_clusters,
                         clustering_type=clustering_type, filter_mode=filter_mode)

    
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

class PMFG(GeneralGraph):

    def __init__(self,
                 return_df: pd.DataFrame,
                 num_clusters: int = 10,
                 clustering_type: str = None,
                 filter_mode: int = None
                 ) -> None:
        super().__init__(return_df, num_clusters,
                         clustering_type=clustering_type, filter_mode=filter_mode)

    
    def build_graph(self) -> nx.Graph:
        """ 
        similar to MST, but further add edges provided that the planarity is maintained.

        Reference: https://gmarti.gitlab.io/networks/2018/06/03/pmfg-algorithm.html
        """
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
        similarities_np[np.isnan(similarities_np)] = - 999  # let nan be the least ones
        sorted_pair_idx = upper_idx_tuple[np.argsort(similarities_np)[::-1]]
        sorted_pair_stock_code = [(tickers[i1], tickers[i2]) for i1, i2 in sorted_pair_idx]

        # count na
        na_stock_counts = (self.similarity_df.isna().mean() > cfg.na_threshold).sum()

        # add edges 
        logging.warn(
            'For performance issue, we adjust number of nodes on PMFG to be less than 3(N - 2)'
        )
        for node_1, node_2 in sorted_pair_stock_code:
            g.add_edge(node_1, node_2)
            if not planarity.is_planar(g): 
                g.remove_edge(node_1, node_2)
            
            # early stopping # * different from 3(N - 2)
            if g.number_of_edges() == int(2.7 * (num_tickers - 2 - na_stock_counts)): # cut na stocks
                break 

        return g

# class RMT(GeneralGraph):
    
#     def __init__(self, return_df: pd.DataFrame, num_clusters: int = 10) -> None:
#         super().__init__(return_df, num_clusters)
#         self.is_graph = False  # for RMT 


