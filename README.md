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


To reproduce the result for Enron dataset, run
```
python src/temporal_structural_walk/run_embedding.py -p data/hospital -d 32 -r 20 -l 25 -w 10 -k 5 --alpha 0.05 --class_type multiclass
```

To reproduce the result for Workplace dataset, run
```
python src/temporal_structural_walk/run_embedding.py -p data/hospital -d 32 -r 20 -l 15 -w 10 -k 5 --alpha 0.025 --class_type multiclass
```

To reproduce the result for Enron dataset, run
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