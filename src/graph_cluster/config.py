"""
Graph Clustering config
"""

# laod packages 
import numpy as np 
import pandas as pd 

from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline

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
