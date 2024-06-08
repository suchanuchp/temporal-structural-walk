# Temporal-Structural Random Walk: Unifying Structural Proximity and Equivalence in Dynamic Networks

## Installation
The code was tested on Python 3.7.11. The requirements are in `setup.py`.
```
cd temporal_structural_walk
pip install .
```

## Usage


1. **-p**, **--datapath** (str, required):
   - Path to the input data file.
   
2. **-s**, **--savedir** (str, optional, default='.'):
   - Directory where results will be saved.
   
3. **-d**, **--embedding_dimension** (int, optional, default=32):
   - Embedding dimensions.
   
4. **-r**, **--random_walks_per_node** (int, optional, default=20):
   - Number of random walks per node.
   
5. **-l**, **--maximum_walk_length** (int, optional, default=20):
   - Maximum length of eac random walk.
   
6. **-w**, **--context_window_size** (int, optional, default=10):
   - Size of the context window for generating embeddings.
   
7. **-k**, **--k** (int, optional, default=-1):
   - Top k neighbors to keep in structural similarity network.
   
8. **-z**, **--save_embeddings** (int, optional, default=0):
   - Flag to save embeddings (0 = no only show the evaluation scores, 1 = yes).
   
9. **--class_type** (str, optional, default='multiclass'):
   - Type of classification problem (`multiclass`, `binary`, `multilabel`).
   
10. **--alpha** (float, optional, default=0.0):
    - Balancing weight in range [0, 1]. When set to 0, the algorithm only considers proximity and the random walk performed is reduced to a temporal walk. When set to 1, the algorithm only considers equivalence, the random walk performed is equivalent to a random walk on a structural similarity network S.


## Dataset Information

| Dataset    | Nodes | Edges   | Time Steps | Classes |
|------------|-------|---------|------------|---------|
| Enron      | 182   | 9,880   | 45         | 7       |
| PPI-aging  | 6,371 | 557,303 | 37         | 2       |
| Brain      | 5,000 | 947,744 | 12         | 10      |

## Node Classification Performance

| Data Set  | Algorithm           | AUPRC             | AUROC             |
|-----------|---------------------|-------------------|-------------------|
| Enron     | DGDV                | 0.3102 ± 0.0216   | 0.6624 ± 0.0171   |
|           | DeepWalk            | 0.4076 ± 0.0433   | 0.7780 ± 0.0279   |
|           | node2vec            | 0.4118 ± 0.1152   | 0.7499 ± 0.0428   |
|           | struc2vec           | 0.3118 ± 0.0597   | 0.6726 ± 0.0460   |
|           | CDNE                | 0.4730 ± 0.0807   | 0.7967 ± 0.0300   |
|           | **Ours (α = 0.10)** | **0.5145 ± 0.0922** | **0.8047 ± 0.0366** |
| PPI-aging | DGDV                | 0.2373 ± 0.0363   | 0.7576 ± 0.0241   |
|           | DeepWalk            | 0.2332 ± 0.0615   | 0.8418 ± 0.0232   |
|           | node2vec            | 0.2375 ± 0.0681   | 0.8296 ± 0.0191   |
|           | struc2vec           | 0.2239 ± 0.0561   | 0.8226 ± 0.0155   |
|           | CDNE                | 0.1035 ± 0.0296   | 0.7592 ± 0.0322   |
|           | **Ours (α = 0.95)**     | **0.2566 ± 0.0311** | 0.7857 ± 0.0262   |
| Brain     | DGDV                | 0.4544 ± 0.0098   | 0.8706 ± 0.0043   |
|           | DeepWalk            | 0.5111 ± 0.0203   | 0.9088 ± 0.0033   |
|           | node2vec            | 0.4956 ± 0.0221   | 0.9063 ± 0.0048   |
|           | struc2vec           | 0.1735 ± 0.0034   | 0.6422 ± 0.0065   |
|           | CDNE                | 0.5571 ± 0.0116   | 0.9200 ± 0.0019   |
|           | **Ours (α = 0.05)**     | **0.5664 ± 0.0156** | **0.9205 ± 0.0027** |

To reproduce the result for Brain dataset, run
```
python src/temporal_structural_walk/run_embedding.py -p data/enron -d 32 -r 20 -l 20 -w 10 -k 5 --alpha 0.1 --class_type multiclass
```

To reproduce the result for PPI-aging dataset, run
```
python src/temporal_structural_walk/run_embedding.py -p data/aging -d 32 -r 20 -l 30 -w 10 -k 100 --alpha 0.95 --class_type binary
```

To reproduce the result for Brain dataset, run
```
python src/temporal_structural_walk/run_embedding.py -p data/brain -d 32 -r 20 -l 10 -w 10 -k 20 --alpha 0.05 --class_type multiclass
```