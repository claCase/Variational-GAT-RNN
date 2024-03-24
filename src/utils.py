import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import networkx as nx
import os
import geopy


def normalize_adj(adj, symmetric=True, power=-0.5, diagonal=False):
    """
    Normalization of Adjacency Matrix
    :param adj (tf.Tensor): (B, N, N)
    :return: adj_normed: (B, N, N)
    """
    if diagonal:
        adj_diag = tf.reduce_mean(adj, -1) * 0.5 + tf.reduce_mean(adj, -2) * 0.5
        adj_diag = tf.linalg.diag(adj_diag)
        adj = adj + adj_diag
    d = tf.reduce_sum(adj, -1)
    d_inv = tf.pow(d, power)
    d_inv = tf.where(tf.math.is_inf(d_inv) | tf.math.is_nan(d_inv), 0.0, d_inv)
    d_inv = tf.linalg.diag(d_inv)
    if symmetric:
        return tf.einsum("...ij, ...jk, ...ko -> ...io", d_inv, adj, d_inv)
    else:
        return tf.einsum("...ij, ...jk -> ...ik", d_inv, adj)
    
    
def node_degree(adj, symmetric=True):
    if symmetric:
        return tf.linalg.diag(tf.reduce_sum(adj, -1))
    else:
        return tf.linalg.diag(tf.reduce_sum(adj, -1), tf.reduce_sum(adj, -2))


def laplacian(adj):
    return node_degree(adj) - adj


def normalized_laplacian(adj, symmetric=True):
    I = tf.linalg.eye(adj.shape[-1], dtype=adj.dtype)[None, :]

    return I - normalize_adj(adj, symmetric)


def add_identity(adj):
    I = tf.linalg.eye(adj.shape[-1])[None, :]
    return adj + I


def power_adj_matrix(adj, power):
    powers = np.empty((power, *adj.shape))
    for i in range(power):
        powers[i, :, :] = np.linalg.matrix_power(adj, i)
    return powers


def laplacian_eigenvectors(adj, type="normalized", **kwargs):
    assert type in {"normalized", "standard"}
    if "symmetric" in kwargs.keys():
        symmetric = kwargs.symmetric
    else:
        symmetric = None

    if type == "normalized":
        lap = normalized_laplacian(adj, symmetric)
    else:
        lap = laplacian(adj)
    if isinstance(adj, tf.Tensor):
        vec, val = tf.linalg.eig(lap)
        return vec, val, lap
    if isinstance(adj, np.ndarray):
        val, vec = np.linalg.eig(lap)
        return val[0], vec[0], lap.numpy()


def outer_eigenvectors(eig):
    if isinstance(eig, tf.Tensor):
        outer = tf.einsum("...ki,...zi->...kzi", eig, eig)
        outer = tf.transpose(outer, perm=(1, 2, 0))
    elif isinstance(eig, np.ndarray):
        outer = np.einsum("...ki,...zi->...kzi", eig, eig)
        outer = np.swapaxes(outer, 2, 0)
    else:
        raise TypeError(
            f"Eigenvectors matrix of type {type(eig)} in not in types (tf.Tensor, np.ndarray)"
        )

    return outer


def diff_log(matrix: np.array):
    t0 = np.nan_to_num(np.log(matrix[:-1]), nan=0.0, neginf=0.0)
    t1 = np.nan_to_num(np.log(matrix[1:]), nan=0.0, neginf=0.0)
    diff = t1 - t0
    return diff


def filter_value(matrix: np.array, value: float, axis=-1):
    filter_ = ~(matrix == value).all(axis=axis)
    return matrix[filter_], filter_


def positive_variance(var):
    return tf.math.maximum(tf.nn.softplus(var), tf.math.sqrt(tf.keras.backend.epsilon()))


def zero_inflated_lognormal(logits=None, p=None, mu=None, sigma=None):
    """
    logits: TxNxNx3
    """
    if logits is not None:
        p, mu, sigma = tf.unstack(logits, axis=-1)
        p = tf.nn.sigmoid(p)
        sigma = positive_variance(sigma)
    perm_axis = tf.concat((tf.range(len(p.shape)) + 1, (0,)), axis=0)
    p_mix = tf.transpose([1 - p, p], perm=perm_axis)
    a = tfd.Mixture(
        cat=tfd.Categorical(probs=p_mix),
        components=[
            tfd.Deterministic(loc=tf.zeros_like(p)),
            tfd.LogNormal(loc=mu, scale=sigma),
        ])
    return a


def get_product_subgraph(G: nx.MultiGraph, prod_key: str) -> nx.Graph:
    edges = [(i, j, k) for i, j, k in G.edges if k == prod_key]
    G_sub = G.edge_subgraph(edges).copy()
    return G_sub


def from_data_sparse_to_ntx(data_sp):
    """
    Parameters:
        data_sp: tf.SparseTensor of shape RxNxN
    """
    edges = data_sp.indices
    r = edges[:, 0]
    i = edges[:, 1]
    j = edges[:, 2]
    idx2prod = idx_to_product(r)
    idx2country_i = idx_to_countries(i)
    idx2country_j = idx_to_countries(j)
    data = pd.DataFrame(
        {
            "ReporterISO3": idx2country_i,
            "PartnerISO3": idx2country_j,
            "ProductCode2": idx2prod,
        }
    )
    G = nx.from_pandas_edgelist(
        df=data[["ReporterISO3", "PartnerISO3", "TradeValue", "ProductCode2"]],
        source="ReporterISO3",
        target="PartnerISO3",
        edge_attr=["TradeValue"],
        edge_key="ProductCode2",
        create_using=nx.MultiDiGraph(),
    )
    return G


def from_edgelist_to_pd(edgelist, values):
    df_convert = pd.read_excel(COUNTRIES_CODES_PATH)
    edgelist = np.asarray(edgelist)
    for i in range(len(edgelist)):
        c1 = df_convert[df_convert["Country Code"] == edgelist[i, 0]][
            "ISO3-digit Alpha"
        ]
        c2 = df_convert[df_convert["Country Code"] == edgelist[i, 1]][
            "ISO3-digit Alpha"
        ]
        edgelist[i, 0] = c1
        edgelist[i, 1] = c2

    for i in range(len(edgelist)):
        edgelist[i, 0] = df_convert[edgelist[i, 0]]
        edgelist[i, 1] = df_convert[edgelist[i, 1]]

    df = pd.DataFrame(
        {
            "code1": edgelist[:, 0],
            "code2": edgelist[:1],
            "prod": edgelist[:, 2],
            "tv": values,
        }
    )
    G = nx.from_pandas_edgelist(
        df=df,
        source="code1",
        target="code2",
        edge_attr=["tv"],
        edge_key="prod",
        create_using=nx.MultiDiGraph(),
    )

    return G


def select_edges(
        edge_list: [(int, int, str)],
        values: [int],
        years: [int],
        code1: [str],
        code2: [str],
        products: [str],
):
    subgraph_edgelist = []
    subgraph_values = []
    for row, tv in zip(edge_list, values):
        y = row[0]
        c1 = row[1]
        c2 = row[2]
        p = row[3]
        if y in years and c1 in code1 and c2 in code2 and p in products:
            subgraph_edgelist.append((y, c1, c2, p))
            subgraph_values.append(tv)
    return subgraph_edgelist, subgraph_values


def idx_to_product(data=None, reporting_code="SITC1"):
    if os.path.exists(
            os.path.join(COMTRADE_DATASET, f"idx_to_prod_{reporting_code}.pkl")
    ):
        with open(
                os.path.join(COMTRADE_DATASET, f"idx_to_prod_{reporting_code}.pkl"), "rb"
        ) as file:
            idx_to_prod = pkl.load(file)
    else:
        prod_to_idx = products_to_idx(reporting_code)
        idx_to_prod = {}
        for k in prod_to_idx.keys():
            idx_to_prod[prod_to_idx[k]] = k
        with open(
                os.path.join(COMTRADE_DATASET, f"idx_to_prod_{reporting_code}.pkl"), "wb"
        ) as file:
            pkl.dump(idx_to_prod, file)
    idxs = []
    if data is not None:
        for i in data:
            idxs.append(idx_to_prod[i])
        return idxs
    else:
        return idx_to_prod


def data_countries_to_idx(data, dict, data_path=None, conversion_dict_path=None):
    if data_path is not None:
        with open(data_path, "rb") as file:
            data = pkl.load(file)
    else:
        data = data

    if conversion_dict_path is not None:
        with open(conversion_dict_path, "rb") as file:
            idx_conversion = pkl.load(file)
    else:
        idx_conversion = dict

    no_key = []
    counter = tqdm.tqdm(total=len(data), desc="Converting countries codes")
    for i, e in enumerate(data):
        try:
            c1 = idx_conversion[e[1]]
            c2 = idx_conversion[e[2]]
            data[i] = (e[0], c1, c2, *e[3:])
        except:
            print(f"no key for {e}")
            no_key.append(i)
        counter.update(1)

    for i in no_key:
        data.pop(i)
    return data


def idx_to_countries(data=None):
    if os.path.exists(os.path.join(COMTRADE_DATASET, "idx_to_countries.pkl")):
        with open(os.path.join(COMTRADE_DATASET, "idx_to_countries.pkl"), "rb") as file:
            idx_to_country = pkl.load(file)
    else:
        country_to_idx = countries_to_idx()
        idx_to_country = {}
        for k in country_to_idx.keys():
            idx_to_country[country_to_idx[k]] = k
            with open(
                    os.path.join(COMTRADE_DATASET, "idx_to_countries.pkl"), "wb"
            ) as file:
                pkl.dump(idx_to_country, file)
    converted = []
    if data is not None:
        for i in data:
            converted.append(idx_to_country[i])
        return converted
    else:
        return idx_to_country



"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import networkx as nx
import zipfile
PATH = "A:\\Users\\Claudio\\Documents\Data\\Networks\\Dynamic"
import os 
mapping_path = os.path.join(PATH, "BACI_HS92_V202301.zip")
zipped_folder = zipfile.ZipFile(mapping_path, "r")
country_codes = zipped_folder.read("country_codes_V202301.csv")
import pandas as pd 
c_codes = pd.DataFrame([x.split(",") for x in country_codes.decode("utf8").replace("\r", "").replace('"', "").split("\n")])
from geopy import distance
distance_graph = nx.DiGraph()
for _, s in c_codes.iso_2digit_alpha.items():
    for _, t in c_codes.iso_2digit_alpha.items():
        if t != s:
            try:
                s_lat_long = long_lat[long_lat.country == s][["latitude", "longitude"]].to_numpy()[0]
                t_lat_long = long_lat[long_lat.country == t][["latitude", "longitude"]].to_numpy()[0]
                distance_graph.add_edge(s, t, weight=distance.geodesic(s_lat_long, t_lat_long).km)    
            except:
                pass 
            
"""
