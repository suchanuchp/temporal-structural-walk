import os
import argparse
import numpy as np
import pandas as pd

from stellargraph import StellarGraph
from random_walk import TemporalStructuralRandomWalk
from gensim.models import Word2Vec

from utils import (
    keep_top_k, euclidean_similarity_matrix,
    train_multiclass, train_multilabel
)

OPT_CRITERIA = 'Average Precision Score (Macro)'
CRITERIA_STAT = 'mean'


def sort_embeddings(model, embedding_dim, nodes_st):
    n_nodes = len(nodes_st)
    embeddings = np.zeros((n_nodes, embedding_dim))
    for i, st in enumerate(nodes_st):
        try:
            embeddings[i] = model.wv[st]
        except KeyError:
            pass
    return embeddings


def main():
    # Check if filename is provided
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--datapath', type=str)
    parser.add_argument('-s', '--savedir', type=str, default='.')
    parser.add_argument('-d', '--embedding_dimension', type=int, default=32)
    parser.add_argument('-r', '--random_walks_per_node', type=int, default=20)
    parser.add_argument('-l', '--maximum_walk_length', type=int, default=20)
    parser.add_argument('-w', '--context_window_size', type=int, default=10)
    parser.add_argument('-k', '--k', type=int, default=-1)
    parser.add_argument('-z', '--save_embeddings', type=int, default=0)
    parser.add_argument('--class_type', type=str, default='multiclass')
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--all', type=int, default=0)

    args = parser.parse_args()
    opt = vars(args)

    data_path = opt['datapath']
    save_dir = opt['savedir']
    embedding_dim = opt['embedding_dimension']
    num_walks_per_node = opt['random_walks_per_node']
    walk_length = opt['maximum_walk_length']
    window_size = opt['context_window_size']
    class_type = opt['class_type']

    if not os.path.exists(data_path):
        raise FileExistsError("data path does not exist")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # raise FileExistsError("data save path does not exist")

    with open(os.path.join(save_dir, 'params.txt'), "w") as f:
        for key, value in opt.items():
            print("{}: {}\n".format(key, value))
            f.write("{}: {}\n".format(key, value))

    df = pd.read_csv(os.path.join(data_path, 'network.txt'), sep=',', names=['src', 'dst', 't', 'w'])

    df.src = df.src.astype(str)
    df.dst = df.dst.astype(str)
    df.head()

    nodes_int = sorted([int(st) for st in set(df.src).union(set(df.dst))])
    nodes_st = [str(node) for node in nodes_int]

    num_cw = len(nodes_st) * num_walks_per_node * (walk_length - window_size + 1)

    dynamic_graph = StellarGraph(
        nodes=pd.DataFrame(index=nodes_st),
        edges=df,
        source_column='src',
        target_column='dst',
        edge_weight_column='t',
    )

    df_label = pd.read_csv(os.path.join(data_path, 'df_label.csv'), index_col=0)
    label = df_label.to_numpy()
    assert label.shape[0] == len(nodes_int)
    if opt['save_embeddings']:
        np.savetxt(os.path.join(save_dir, f'label.txt'), label)

    run_cross_walk(dynamic_graph, opt=opt, label=label, nodes_st=nodes_st)


def run_cross_walk(ppi_graph, opt, label, nodes_st):
    np.random.seed(0)
    embedding_dim = opt['embedding_dimension']
    num_walks_per_node = opt['random_walks_per_node']
    walk_length = opt['maximum_walk_length']
    window_size = opt['context_window_size']
    save_dir = opt['savedir']
    alpha = opt['alpha']
    num_cw = len(nodes_st) * num_walks_per_node * (walk_length - window_size + 1)
    structural_graph = get_structural_sim_network(ppi_graph, nodes_st, opt)
    cross_temporal_rw = TemporalStructuralRandomWalk(ppi_graph, structural_graph=structural_graph)
    cross_walks = cross_temporal_rw.run(
        num_cw=num_cw,
        cw_size=window_size,
        max_walk_length=walk_length,
        walk_bias="exponential",
        seed=0,
        alpha=alpha,
    )

    cross_walk_model = Word2Vec(
        cross_walks,
        vector_size=embedding_dim,
        window=window_size,
        min_count=0,
        negative=10,
        sg=1,
        workers=2,
        seed=0
    )
    embeddings = sort_embeddings(cross_walk_model, embedding_dim, nodes_st)
    print(f'-------alpha: {alpha}-------')
    result = train(embeddings, label, class_type=opt['class_type'], to_print=True)
    if opt['save_embeddings']:
        np.savetxt(os.path.join(save_dir, f'embedding_alpha_{alpha}.txt'), embeddings)

    return result


def get_structural_sim_network(ppi_graph, nodes_st, opt):
    from sklearn.preprocessing import StandardScaler
    data_path = opt['datapath']

    import networkx as nx
    if os.path.exists(os.path.join(data_path, 'dynamic_graphlets/sorted_output_dgdv_6_4_1.txt')):
        dgdv = np.loadtxt(os.path.join(data_path, 'dynamic_graphlets/sorted_output_dgdv_6_4_1.txt'))
    else:
        try:
            dgdv = np.loadtxt(os.path.join(data_path, 'dynamic_graphlets/sorted_output_dgdv_4_4_1.txt'))
        except Exception:
            dgdv = np.loadtxt(os.path.join(data_path, 'dynamic_graphlets/sorted_output_dgdv_4_5_1.txt'))
    scaler = StandardScaler()
    scaled_dgdv = scaler.fit_transform(dgdv)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.9, random_state=0)
    scaled_dgdv = pca.fit_transform(scaled_dgdv)
    sim = euclidean_similarity_matrix(scaled_dgdv)

    np.fill_diagonal(sim, 0)
    if opt['k'] == -1:
        k = int(np.mean(list(ppi_graph.node_degrees().values())))
    else:
        k = opt['k']
    sim = keep_top_k(sim, k)
    df_sim = pd.DataFrame(sim, index=nodes_st, columns=nodes_st)
    g_sim = nx.from_pandas_adjacency(df_sim)
    g_sim = StellarGraph.from_networkx(g_sim)
    return g_sim


def train(embeddings, label, class_type, to_print=True):
    if class_type == 'multilabel':
        return train_multilabel(embeddings, label, to_print=to_print)
    else:
        return train_multiclass(embeddings, label, to_print=to_print)


if __name__ == "__main__":
    main()
