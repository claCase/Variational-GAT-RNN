from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import networkx as nx
import os
import geopy
import pickle as pkl


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
    return tf.math.maximum(
        tf.nn.softplus(var), tf.math.sqrt(tf.keras.backend.epsilon())
    )


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
        ],
    )
    return a


def get_product_subgraph(G: nx.MultiGraph, prod_keys: List[str]) -> nx.Graph:
    edges = [(i, j, k) for i, j, k in list(G.edges) if k in prod_keys]
    G_sub = G.edge_subgraph(edges).copy()
    return G_sub


def country_code_converter(countries:List[Union[str, int]], iso="iso2"):
    if len(countries) == 0:
        return countries
    if type(countries[0]) is int:
        from_iso = "country_code"
    elif type(countries[0]) is str:
        if len(countries[0]) == 2:
            from_iso = "iso_2digit_alpha"
        elif len(countries[0]) == 3:
            from_iso = "iso_3digit_alpha"
        elif len(countries[0]) > 3:
            from_iso = "country_name_abbreviation"
        else:
            raise ValueError(
                "country_code_converter: Country code is of type string but is not of length 2 or 3"
            )
    else:
        raise ValueError("country_code_converter: Country code is not of type string or int")

    if iso == "iso2":
        to_iso = "iso_2digit_alpha"
    elif iso == "iso3":
        to_iso = "iso_3digit_alpha"
    elif iso == "code":
        to_iso = "country_code"
    elif iso == "name":
        to_iso = "country_name_abbreviation"
    else:
        raise ValueError("country_code_converter: iso must be (in iso2, iso3, code, name)")
    if from_iso == to_iso:
        return countries
    else:
        cc = pd.read_csv("./data/country_codes_V202301.csv")
        return [cc[cc[from_iso] == name][to_iso].values[0] for name in countries]


def country_long_lat(countries):
    with open(os.path.join(os.getcwd(), "Data", "iso2_long_lat.pkl"), "rb") as file:
        lat_long = pkl.load(file)

    countries = country_code_converter(countries, "iso2")
    cc = pd.read_csv("./data/country_codes_V202301.csv")

    return [lat_long[name] for name in countries]


# https://community.plotly.com/t/scattermapbox-plot-curved-lines-like-scattergeo/43665
def point_sphere(lon, lat):
    #associate the cartesian coords (x, y, z) to a point on the  globe of given lon and lat
    #lon longitude
    #lat latitude
    lon = lon*np.pi/180
    lat = lat*np.pi/180
    x = np.cos(lon) * np.cos(lat) 
    y = np.sin(lon) * np.cos(lat) 
    z = np.sin(lat) 
    return np.array([x, y, z])


def slerp(A=[100, 45], B=[-50, -25], dir=-1, n=100):
    #Spherical "linear" interpolation
    """
    A=[lonA, latA] lon lat given in degrees; lon in  (-180, 180], lat in ([-90, 90])
    B=[lonB, latB]
    returns n points on the great circle of the globe that passes through the  points A, B
    #represented by lon and lat
    #if dir=1 it returns the shortest path; for dir=-1 the complement of the shortest path
    """
    As = point_sphere(A[0], A[1])
    Bs = point_sphere(B[0], B[1])
    alpha = np.arccos(np.dot(As,Bs)) if dir==1 else  2*np.pi-np.arccos(np.dot(As,Bs))
    if abs(alpha) < 1e-6 or abs(alpha-2*np.pi)<1e-6:
        return A
    else:
        t = np.linspace(0, 1, n)
        P = np.sin((1 - t)*alpha) 
        Q = np.sin(t*alpha)
        #pts records the cartesian coordinates of the points on the chosen path
        pts =  np.array([a*As + b*Bs for (a, b) in zip(P,Q)])/np.sin(alpha)
        #convert cartesian coords to lons and lats to be recognized by go.Scattergeo
        lons = 180*np.arctan2(pts[:, 1], pts[:, 0])/np.pi
        lats = 180*np.arctan(pts[:, 2]/np.sqrt(pts[:, 0]**2+pts[:,1]**2))/np.pi
        return lons, lats
    

def select_edges(
    edge_list: List[Tuple[int, int, str]],
    values: List[int],
    years: List[int],
    code1: List[str],
    code2: List[str],
    products: List[str],
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
