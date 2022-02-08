"""
Graph Clustering config
"""

# =======================
#        metrics 
# =======================

# pick a similarity measure
similarity_metric = 'cor'

# ======================
#         paths 
# ======================

# fig plot path 
fig_save_path = 'out/graph_fig/'

# discard points with 60% nan in a single correlation period
na_threshold = 0.6

# ===========================
#       graph specifics 
# ===========================

# supporting AG, MST, and PMFG 
graph_type = 'MST'
num_clusters = 10 

 # supporting single_linkage, spectral, node2vec, sub2vec, 
clustering_type = 'spectral'   

# filter information {0: raw, 1: keep large eigenvalues; 2: keep large but market mode}
filter_mode = 2

# =============================
#        stock related       
# =============================

# support sz50, hs300, zz500, zz1000. Mostly on zz1000
stock_pool = 'zz1000'

# =======================================
# ------ community detection utils ------
# =======================================

# 1. Node2Vec
# for random walks
node2vec_rw_params = {
    'length': 10,   # maximum length of a random walk
    'n': 10,        # number of random walks per root node
    'p': 0.5,       # Defines (unormalised) probability, 1/p, of returning to source node
    'q': 2.0        # Defines (unormalised) probability, 1/q, for moving away from source node
}
# for word2vec
node2vec_word2vec_params = {
    'vector_size': 20,  # embedding dimensions
    'window': 5,        # maximum distance between a word and the prediction
    'min_count': 0,     # ignore all words below this frequency
    'sg': 1,            # 1 for skip-gram, 0 for CBOW
    'epochs': 1         # number of times passing through the dataset
}


# 2. Sub2Vec
sub2vec_walk_length =  5   # length of the random walk
num_hops = 2               # number of hops for neighbors
sub2vec_mode = 3           # fyi 1: neighbor; 2: structural; 3: concat both
sub2vec_params = {
    'vector_size': 10,  # embedding dimensions 
    'window': 5,        # maximum distance between a word and the prediction
    'dm': 1,            # 1 for dm and 0 for DBOW
    'epochs': 1,        # number of times passing through the dataset
    'min_count': 1      # ignore all words below this threshold
}


# ======================================
# ------- for developers ---------------
# ======================================

temp_graph_save_path = 'src/graph_cluster'
