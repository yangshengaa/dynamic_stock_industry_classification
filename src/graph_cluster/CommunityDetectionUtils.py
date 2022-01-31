"""
utils for complex community detection algorithms, including
- Node2Vec
- Sub2Vec
"""

# load packages 
import os
import time
import pickle
import random
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count
from functools import partial
from typing import Dict, List

import networkx as nx 

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.cluster import KMeans

# load config 
import src.graph_cluster.config as cfg


# ================================================
# ---------------- Node2Vec ----------------------
# ================================================

class Node2Vec:

    def __init__(self, g: nx.Graph, num_clusters: int=10) -> None:
        self.g = g
        self.num_clusters = num_clusters

        # read config 
        # for random walks 
        self.node2vec_rw_params = cfg.node2vec_rw_params
        # for word2vec 
        self.node2vec_word2vec_params = cfg.node2vec_word2vec_params

    def generate_embeddings(self): 
        """ get Node2Vec embeddings """
        # convert to stellar graph 
        g = StellarGraph.from_networkx(self.g)

        # generate random walks 
        rw = BiasedRandomWalk(g)
        walks = rw.run(
            nodes=list(g.nodes()),  # root nodes
            **self.node2vec_rw_params
        )

        # compute embeddings
        str_walks = [[str(n) for n in walk] for walk in walks]
        model = Word2Vec(
            str_walks, 
            workers=cpu_count(),
            **self.node2vec_word2vec_params 
        )

        # get embeddings 
        embeddings = np.vstack([model.wv.get_vector(key) for key in self.g.nodes])
        self.embeddings = embeddings

    def get_community(self) -> Dict[str, int]: 
        """ get the community label from embeddings """
        labels = KMeans(n_clusters=self.num_clusters).fit_predict(self.embeddings)
        label_dict = dict(zip(self.g.nodes, labels))
        return label_dict

# ================================================
# ---------------- Node2Vec ----------------------
# ================================================

class Sub2Vec:

    def __init__(self, g: nx.Graph, num_clusters: int = 10) -> None:
        self.g = g
        self.num_clusters = num_clusters
        # read config 
        self.sub2vec_params = cfg.sub2vec_params
        self.walk_length = cfg.sub2vec_walk_length
        self.k = cfg.num_hops 
        self.sub2vec_mode = cfg.sub2vec_mode
        assert self.sub2vec_mode in [1, 2, 3], 'sub2vec mode not implemented'

        # for developer 
        self.temp_graph_save_path = cfg.temp_graph_save_path

    def arr2str(self, arr):
        """ convert array to string, separated by spaces """
        result = ""
        for i in arr:
            result += " " + str(i)
        return result

    def generate_embeddings(self): 
        """ generate embeddings """
        # get rws and embeddings
        if self.sub2vec_mode == 1 or self.sub2vec_mode == 3:
            neighbor_rws = self.generate_neightor_random_walks()
            neighbor_embeddings = self.get_doc2vec_embeddings(neighbor_rws)
        if self.sub2vec_mode == 2 or self.sub2vec_mode == 3:
            structural_rws = self.generate_structural_random_walks()
            structural_embeddings = self.get_doc2vec_embeddings(structural_rws)

        # return embeddings
        if self.sub2vec_mode == 1:
            self.embeddings = neighbor_embeddings
        elif self.sub2vec_mode == 2:
            self.embeddings = structural_embeddings
        elif self.sub2vec_mode == 3:
            self.embeddings = np.hstack([neighbor_embeddings, structural_embeddings])


    def get_community(self) -> Dict[str, int]:
        """ get the community label by kmeans """
        labels = KMeans(n_clusters=self.num_clusters).fit_predict(self.embeddings)
        label_dict = dict(zip(self.g.nodes, labels))
        return label_dict

    # ------------- auxiliary -------------
    def get_doc2vec_embeddings(self, rws: List[TaggedDocument]) -> np.ndarray:
        """ 
        from random walks, generate subgraph embeddings
        """
        model = Doc2Vec(documents=rws, **self.sub2vec_params)
        doc_vectors = np.vstack([model.docvecs.get_vector(tag) for tag in self.g.nodes])
        return doc_vectors
    
    @staticmethod
    def k_neighbors(g: nx.Graph, source: str or int, k: int = 1) -> List[str or int]:
        """ 
        return k-th hop neighbors, including source 
        :param source: the source node 
        :return a list of nodes 
        """ 
        raw_dict = nx.single_source_shortest_path_length(g, source, k)
        neighbors = list(raw_dict.keys())
        return neighbors

    def in_range(self, val: float) -> str:
        """ get the label for the range """
        # in original implementations 
        range_to_label = {
            (0, 0.05): 'z',
            (0.05, 0.1): 'a',
            (0.1, 0.15): 'b',
            (0.15, 0.2): 'c',
            (0.2, 0.25): 'd',
            (0.25, 0.5): 'e',
            (0.5, 0.75): 'f',
            (0.75, 1.0): 'g'
        }
        # search for the tag
        for key in range_to_label:
            if key[0] <= val and key[1] > val:  # left close, right open
                return range_to_label[key]
    
    # -------- random walks for two different channels ----------------
    # 1. neighbors
    def generate_neightor_random_walks(self) -> List[TaggedDocument]:
        """ 
        generate random walks (id-paths) 
        :return a random walk (in tagged documents) in a list 
        """
        tagged_document_list = []

        # # save local 
        # g_path = os.path.join(self.temp_graph_save_path, 'temp_graph')
        # with open(g_path, 'wb') as f:
        #     pickle.dump(self.g, f)
        
        # # mp 
        # partial_func = partial(
        #     Sub2Vec.neighbor_rw_sampler, 
        #     k=self.k, 
        #     walk_length=self.walk_length,
        #     g_path=g_path
        # )
        # pool = mp.Pool()
        # results = pool.map(partial_func, list(self.g.nodes), chunksize=200)
        # pool.close()
        # pool.join()

        # # convert to tag 
        # for result, tag in zip(results, self.g.nodes):
        #     tagged_document_list.append(TaggedDocument(result, [tag]))
        
        # # remove temp graph
        # os.remove(g_path)
        
        # return tagged_document_list

        # loop through each node 
        for node in self.g.nodes:
            walk_list = []
            cur_node = node

            # extract subgraph (k-hop neighbors)
            cur_node_neighbors = Sub2Vec.k_neighbors(self.g, cur_node, k=self.k)
            sub_graph = self.g.subgraph(cur_node_neighbors)

            if sub_graph.number_of_edges() > 0:
                # random walk on subgraphs
                while(len(walk_list) < self.walk_length):
                    walk_list.append(cur_node)
                    cur_node = random.choice(list(sub_graph.neighbors(cur_node)))
            else:  # unconnected nodes
                walk_list = [cur_node]
            
            # convert to tagged document
            tagged_document_list.append(TaggedDocument(walk_list, [node]))
        return tagged_document_list
    
    # @staticmethod
    # def neighbor_rw_sampler(node, k, walk_length, g_path) -> List[str or int]:
    #     """ subprocess of a random walk sampler """
    #     # read from file 
    #     with open(g_path, 'rb') as f:
    #         g = pickle.load(f)
        
    #     # extract subgraph
    #     walk_list = []
    #     cur_node = node
    #     raw_dict = nx.single_source_shortest_path_length(g, node, k)
    #     cur_node_neighbors = list(raw_dict.keys())
    #     sub_graph = g.subgraph(cur_node_neighbors)

    #     # random walk
    #     if sub_graph.number_of_edges() > 0:
    #         # random walk on subgraphs
    #         while(len(walk_list) < walk_length):
    #             walk_list.append(cur_node)
    #             cur_node = random.choice(list(sub_graph.neighbors(cur_node)))
    #     else:  # unconnected nodes
    #         walk_list = [node]
    #     return walk_list

    # 2. structural
    def generate_structural_random_walks(self) -> List[TaggedDocument]:
        """ 
        generate random walks (degree-paths)
        :return a random walk (in tagged documents) in a list 
        """
        # compute degrees
        degree_dict = dict(self.g.degree(self.g.nodes()))
        label_dict = {}
        total_num_nodes = float(self.g.number_of_nodes())
        for node in degree_dict.keys():
            val = degree_dict[node] / total_num_nodes
            label_dict[node] = {'label': self.in_range(val)}
        nx.set_node_attributes(self.g, label_dict)

        # generat walk
        tagged_document_list = []

        # loop through each node
        for node in self.g.nodes:
            walk_list = []
            cur_node = node

            # extract subgraph (k-hop neighbors)
            cur_node_neighbors = Sub2Vec.k_neighbors(self.g, cur_node, k=self.k)
            sub_graph = self.g.subgraph(cur_node_neighbors)

            if sub_graph.number_of_edges() > 0:
                # random walk on subgraphs
                while(len(walk_list) < self.walk_length):
                    walk_list.append(self.g.nodes[cur_node]['label'])  # extract label only 
                    cur_node = random.choice(list(sub_graph.neighbors(cur_node)))
            else: # for unconnected nodes
                walk_list = [cur_node]

            # convert to tagged document
            tagged_document_list.append(TaggedDocument(walk_list, [node]))

        return tagged_document_list
