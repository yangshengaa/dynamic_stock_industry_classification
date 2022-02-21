# Graph Cluster

Experiment different graph clustering methodologies. The following graphs are to be implemented:

- Asset Graph (AG)
- Minimum Spanning Tree (MST)
- Planar Maximally Filtered Graph (PMFG)

In addition to these, each of the first three could be built upon correlation matrices transformed using techniques covered in RMT.

For each constructed graph, we could use four different ways to detect communities:

- Average Linkage
- Spectral Clustering
- Node2Vec + KMeans
- Sub2Vec + KMeans

## Running Instructions

The main module is to train industry. There are two versions: ```IndustryTrainer``` and ```MultiIndustryTrainer```. The former one train one set of parameters, whereas the latter one obtain communities by different sets of parameters when the graph is constructed in each period.

```bash
# read config to train one set of parameter
python run.py cluster_train 

# use combinations of these config to train multiple communities
python run.py cluster_train --use_multi True --graph_type AG --filter_mode 0 --multi_num_clusters 5 10 20 30 40 60 --multi_clustering_type spectral node2vec sub2vec 
```

## Demo

MST: zz1000, trained using the entire 2021.

![mst_graph](../../out/graph_fig/mst_20210104_20211231_zz1000.png)
