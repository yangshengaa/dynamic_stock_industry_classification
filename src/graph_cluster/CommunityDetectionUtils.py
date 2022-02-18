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
import numbers
import logging
import numpy as np
import pandas as pd

import multiprocessing as mp
from multiprocessing import cpu_count
from functools import partial

from typing import Dict, Iterable, List, Type

import networkx as nx 

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.cluster import KMeans

# load config 
import src.graph_cluster.config as cfg

# constants 
EPS = 1e-6       # stop criterion for updating modularity
MAX_PASS = - 1 # -1 means no restriction on number of pass 

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
# ---------------- Sub2Vec -----------------------
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

# ================================================
# -------------------- RMT -----------------------
# ================================================

# Modified Louvain
# the followings are modified from https://github.com/taynaud/python-louvain
# * impractical except the information filtering part ...

class LouvainGraphStatus(object):
    """
    Louvain Graph handler: 
    To handle several data in one struct.
    Could be replaced by named tuple, but don't want to depend on python 2.6
    """

    def __init__(self):
        self.node2com = {}
        self.total_weight = 0
        self.degrees = {}
        self.gdegrees = {}
        self.internals = {}
        self.loops = {}

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = LouvainGraphStatus()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = {}
        self.total_weight = 0
        self.degrees = {}
        self.gdegrees = {}
        self.internals = {}
        self.loops = {}
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc

class ModifiedLouvain:

    def __init__(self, similarity_df: pd.DataFrame, num_clusters: int = 10) -> None:
        self.similarity_df = similarity_df 
        self.num_clusters = num_clusters
        # load from config
        self.filter_mode = cfg.filter_mode

    @staticmethod
    def filter_information(
            similarity_df: pd.DataFrame, 
            filter_mode: int,
            T: int=240
        ) -> pd.DataFrame:
        """ 
        filter information: 
        - 0: filtered self-loops (raw)
        - 1: keep large eigenvalues only 
        - 2: keep large eigenvalues except the market mode 

        on top of these nan values are processed (keep as a mask)

        :param similarity_df: correlation matrix, for instance 
        :param filter_mode: 0, 1, or 2, using the rule defined above;
        :param T: the number of days used to obtain the similarity_df

        :return filtered correlation information in a pandas dataframe
        """
        filtered_df = None
        # filter information according to mode
        if filter_mode == 0:
            filtered_df = similarity_df
        else:
            # create mask
            na_pos = similarity_df.notna().astype(int)
            na_mask = na_pos / na_pos 
            # eigendecomp
            eig_values, eig_vectors = np.linalg.eig(similarity_df.fillna(0))
            N = similarity_df.shape[0]  # number of assets
            if N > T:
                logging.warn(
                    'N > T: statistically unstable in giving the eig value upper bound. Use it anyways'
                )
            eig_value_plus = (1 + np.sqrt(N / T)) ** 2
            max_eig_value = max(eig_values)

            # keep large eig value only 
            filtered_np = 0
            if filter_mode == 1:
                for i in range(N):
                    if eig_values[i] > eig_value_plus:
                        cur_eig_vector = eig_vectors[:, [i]] # keep as a column vector
                        filtered_np = filtered_np + eig_values[i] * cur_eig_vector @ cur_eig_vector.T
            # also eliminate market mode
            elif filter_mode == 2:
                for i in range(N):
                    if eig_values[i] > eig_value_plus and eig_values[i] < max_eig_value:
                        cur_eig_vector = eig_vectors[:, [i]] # keep as a column vector
                        filtered_np = filtered_np + eig_values[i] * cur_eig_vector @ cur_eig_vector.T
            
            # put back into a dataframe 
            filtered_df = pd.DataFrame(
                filtered_np, 
                columns=similarity_df.columns, 
                index=similarity_df.index
            )
            filtered_df = filtered_df * na_mask  # add back mask
            filtered_df = filtered_df.applymap(lambda x: np.real(x))  # keep real for each entry

        return filtered_df
        
    def build_graph_for_louvain(self):
        """ filter information and build graph """
        # ! TODO: add back 
        # filtered_df = ModifiedLouvain.filter_information(self.similarity_df, self.filter_mode)

        # # add nodes
        # g = nx.Graph()
        # tickers  = filtered_df.columns
        # g.add_nodes_from(tickers)

        # # add edges
        # upper_index = np.triu_indices(n=filtered_df.shape[0], k=1)
        # filtered_info_np = filtered_df.values[upper_index]
        # ticker_tuple = [(tickers[i1], tickers[i2]) for i1, i2 in zip(*upper_index)]
        # for (stock_1, stock_2), similarity in zip(ticker_tuple, filtered_info_np):
        #     if not np.isnan(similarity):
        #         g.add_edge(stock_1, stock_2, weight=similarity)

        g = nx.from_numpy_array((self.similarity_df - np.eye(self.similarity_df.shape[0])).values)
        self.g = g

    def get_community(self) -> Dict[int or str, int]:
        """ obtain community in modified louvain """
        # build graph 
        self.build_graph_for_louvain()
        # feed graph to louvain 
        partition = self.best_partition(self.g)
        return partition

    # --------------- louvain ---------------------

    def check_random_state(self, seed):
        """
        Turn seed into a np.random.RandomState instance.
        
        :param seed: None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
        """
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                            " instance" % seed)
    def partition_at_level(
            self, 
            dendrogram: List[Dict[int or str, int or str]], 
            level: int
        ) -> Dict[int or str, int or str]:
        """
        Return the partition of the nodes at the given level
        A dendrogram is a tree and each level is a partition of the graph nodes.
        Level 0 is the first partition, which contains the smallest communities,
        and the best is len(dendrogram) - 1.
        The higher the level is, the bigger are the communities

        :param dendrogram: list of dict, a list of partitions, 
                ie dictionaries where keys of the i+1 are the values of the i.
        :param level: int, the level which belongs to [0..len(dendrogram)-1]

        :return partition: dict a dictionary where keys are the nodes and the values are the set it
        belongs to
        """
        partition = dendrogram[0].copy()
        for index in range(1, level + 1):
            for node, community in partition.items():
                partition[node] = dendrogram[index][community]
        return partition

    def best_partition(
            self,
            graph: nx.Graph,
            partition: Dict[int or str, int or str] = None,
            weight: str = 'weight',
            randomize: bool = True, 
            random_state: np.random.RandomState = None
        ) -> Dict[int or str, int or str]:
        """
        Compute the partition of the graph nodes which maximizes the modularity
        (or try..) using the Louvain heuristics
        This is the partition of highest modularity, i.e. the highest partition
        of the dendrogram generated by the Louvain algorithm.

        :param graph : networkx.Graph, the networkx graph which is decomposed
        :param partition : dict, optional, the algorithm will start using this partition of the nodes.
            It's a dictionary where keys are their nodes and values the communities
        :param weight : str, optional
            the key in graph to use as weight. Default to 'weight'
        :param resolution :  double, optional
            Will change the size of the communities, default to 1.
            represents the time described in
            "Laplacian Dynamics and Multiscale Modular Structure in Networks",
            R. Lambiotte, J.-C. Delvenne, M. Barahona
        :param randomize : boolean, optional
            Will randomize the node evaluation order and the community evaluation
            order to get different partitions at each call
        :param random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        :return partition : dictionnary
        The partition, with communities numbered from 0 to number of communities
        """

        # TODO: only partition up to a certain number of clusters and stop 
        dendo = self.generate_dendrogram(
            graph,
            partition,
            weight,
            randomize,
            random_state
        )
        return self.partition_at_level(dendo, len(dendo) - 1)


    def generate_dendrogram(
            self, 
            graph: nx.Graph,
            part_init: Dict[int or str, int or str] = None,
            weight: str = 'weight',
            randomize: bool = True, 
            random_state: np.random.RandomState = None 
        ) -> List[Dict[int or str, int or str]]:
        """
        Find communities in the graph and return the associated dendrogram
        A dendrogram is a tree and each level is a partition of the graph nodes.
        Level 0 is the first partition, which contains the smallest communities,
        and the best is len(dendrogram) - 1. The higher the level is, the bigger
        are the communities


        :param graph : networkx.Graph, the networkx graph which will be decomposed
        :param part_init : dict, optional the algorithm will start using this partition of the nodes. 
            It's a dictionary where keys are their nodes and values the communities
        :param weight : str, optional, the key in graph to use as weight. Default to 'weight'
        :param resolution :  double, optional, will change the size of the communities, default to 1.
            represents the time described in
            "Laplacian Dynamics and Multiscale Modular Structure in Networks",
            R. Lambiotte, J.-C. Delvenne, M. Barahona

        :return dendrogram : list of dictionaries
            a list of partitions, ie dictionaries where keys of the i+1 are the
            values of the i. and where keys of the first are the nodes of graph
        
        """
        if graph.is_directed():
            raise TypeError("Bad graph type, use only non directed graph")

        if randomize is False:
            random_state = 0

        # We don't know what to do if both `randomize` and `random_state` are defined
        if randomize and random_state is not None:
            raise ValueError(
                "`randomize` and `random_state` cannot be used at the same time"
            )
        
        # init random state
        random_state = self.check_random_state(random_state)   

        # special case, when there is no link
        # the best partition is everyone in its community
        if graph.number_of_edges() == 0:
            part = {}
            for i, node in enumerate(graph.nodes()):
                part[node] = i
            return [part]

        current_graph = graph.copy()
        status = LouvainGraphStatus()
        status.init(current_graph, weight, part_init)

        status_list = list()
        self.__one_level(current_graph, status, weight, random_state)
        new_mod = self.__modularity(status)
        partition = self.__renumber(status.node2com)
        
        status_list.append(partition)
        mod = new_mod
        current_graph = self.induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
        # print(status)

        while True:
            self.__one_level(current_graph, status, weight, random_state)
            new_mod = self.__modularity(status)
            
            if new_mod - mod < EPS:
                break
            partition = self.__renumber(status.node2com)
            status_list.append(partition)

             # early stopping: 
            if len(set(list(partition.values()))) <= self.num_clusters:
                break

            mod = new_mod
            current_graph = self.induced_graph(partition, current_graph, weight)
            status.init(current_graph, weight)
        return status_list[:]


    def induced_graph(
            self,
            partition: Dict[int or str, int or str], 
            graph: nx.Graph, 
            weight: str="weight"
        ):
        """
        Produce the graph where nodes are the communities
        there is a link of weight w between communities if the sum of the weights
        of the links between their elements is w

        :param partition : dict a dictionary where keys are graph nodes and  values the part the node
            belongs to
        :param graph : networkx.Graph, the initial graph
        :param weight : str, optional, the key in graph to use as weight. Default to 'weight'
        
        :return g : networkx.Graph, a networkx graph where nodes are the parts
        """
        ret = nx.Graph()
        ret.add_nodes_from(partition.values())

        for node1, node2, datas in graph.edges(data=True):
            edge_weight = datas.get(weight, 1)
            com1 = partition[node1]
            com2 = partition[node2]
            w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
            ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

        return ret


    def __renumber(self, dictionary: Dict[int or str, int or str]) -> Dict[int or str, int or str]:
        """ Renumber the values of the dictionary from 0 to n """
        values = set(dictionary.values())
        target = set(range(len(values)))

        if values == target:
            # no renumbering necessary
            ret = dictionary.copy()
        else:
            # add the values that won't be renumbered
            renumbering = dict(zip(target.intersection(values),
                                target.intersection(values)))
            # add the values that will be renumbered
            renumbering.update(dict(zip(values.difference(target),
                                        target.difference(values))))
            ret = {k: renumbering[v] for k, v in dictionary.items()}

        return ret

    def __one_level(
            self,
            graph: nx.Graph, 
            status: LouvainGraphStatus, 
            weight_key: str,  
            random_state: int
        ):
        """ Compute one level of communities """
        modified = True
        nb_pass_done = 0
        cur_mod = self.__modularity(status)
        new_mod = cur_mod

        # start epoch
        while modified and nb_pass_done != MAX_PASS:
            cur_mod = new_mod
            modified = False
            nb_pass_done += 1
            
            # loop through each node to connect 
            for node in self.__randomize(graph.nodes(), random_state):
                com_node = status.node2com[node]
                # degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
                neigh_communities = self.__neighcom(node, graph, status, weight_key)

                # print(neigh_communities)
                # remove_cost = - neigh_communities.get(com_node,0) + \
                #     resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
                
                # remove_cost = - neigh_communities.get(com_node,0) + \
                #     1 * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
                # ! TODO: verify this 
                remove_cost = - neigh_communities.get(com_node, 0) 
                # print('remove cost: ', nb_pass_done, remove_cost)
                self.__remove(node, com_node, neigh_communities.get(com_node, 0.), status)
                best_com = com_node
                best_increase = 0
                for com, dnc in self.__randomize(neigh_communities.items(), random_state):
                    # incr = remove_cost + dnc - \
                    #     resolution * status.degrees.get(com, 0.) * degc_totw
                    # ! TODO: verify this 
                    incr = remove_cost + dnc  # * no normalization is needed
                    # incr = remove_cost + dnc - \
                    #     1 * status.degrees.get(com, 0.) * degc_totw
                    if incr > best_increase:
                        best_increase = incr
                        best_com = com
                #         print(best_increase, best_com)
                # print('over')
                # print(node, best_com)
                self.__insert(node, best_com, neigh_communities.get(best_com, 0.), status)
                if best_com != com_node:
                    modified = True
            new_mod = self.__modularity(status)
            print(nb_pass_done, len(set(status.node2com.values())), new_mod)
            if new_mod - cur_mod < EPS:
                break

    def __neighcom(
            self,
            node: str or int, 
            graph: nx.Graph, 
            status: LouvainGraphStatus, 
            weight_key: str
        ) -> Dict[int or str, float]:
        """
        Compute the communities in the neighborhood of node in the graph given
        with the decomposition node2com
        """
        weights = {}
        for neighbor, datas in graph[node].items():
            if neighbor != node:
                edge_weight = datas.get(weight_key, 1)
                neighborcom = status.node2com[neighbor]
                weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

        return weights

    def __remove(
            self,
            node: int or str, 
            com: int or str, 
            weight: float, 
            status: LouvainGraphStatus
        ):
        """ Remove node from community com and modify status """
        status.degrees[com] = (status.degrees.get(com, 0.)
                            - status.gdegrees.get(node, 0.))
        status.internals[com] = float(status.internals.get(com, 0.) -
                                    weight - status.loops.get(node, 0.))
        status.node2com[node] = -1

    def __insert(
            self,
            node: int or str, 
            com: int or str, 
            weight: float, 
            status: LouvainGraphStatus
        ):
        """ Insert node into community and modify status """
        status.node2com[node] = com
        status.degrees[com] = (status.degrees.get(com, 0.) +
                            status.gdegrees.get(node, 0.))
        status.internals[com] = float(status.internals.get(com, 0.) +
                                    weight + status.loops.get(node, 0.))

    def __modularity(
            self, 
            status: LouvainGraphStatus
        ) -> float:
        """
        Fast compute the modularity of the partition of the graph using
        status precomputed
        """
        links = float(status.total_weight)  # C_norm
        result = 0.
        for community in set(status.node2com.values()):
            in_degree = status.internals.get(community, 0)
            degree = status.degrees.get(community, 0)
            if links > 0:
                # result += in_degree * resolution / links -  ((degree / (2. * links)) ** 2)
                result += in_degree / links # ! TODO: verify this
                # result += in_degree / links -  ((degree / (2 * links)) ** 2)
                # result += 0 
        # print(result)
        return result

    def __randomize(
            self,
            items: Iterable, 
            random_state: np.random.RandomState
        ):
        """Returns a List containing a random permutation of items"""
        randomized_items = list(items)
        random_state.shuffle(randomized_items)
        return randomized_items
