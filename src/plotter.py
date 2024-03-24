import tensorflow as tf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_product_subgraph
import os


def draw_subgraph(
        G: nx.MultiDiGraph,
        prod_keys: [str],
        nodes: [str] = None,
        log_scale=True,
        normalize=True,
        quantile=0.10,
):
    with open(os.path.join(os.getcwd(), "Data", "iso2_long_lat.pkl"), "rb") as file:
        lat_long = pkl.load(file)
    radius = np.linspace(-0.1, -0.5, len(prod_keys))
    col_idx = np.arange(0, len(prod_keys))
    colors = plt.get_cmap("Set1")(col_idx)
    ax = plt.axes(projection=ccrs.PlateCarree())
    for i, key in enumerate(prod_keys):
        G_sub = get_product_subgraph(G, key)
        remove_nodes = []
        for name in G_sub.nodes:
            if nodes is not None:
                print(name)
                if name not in nodes:
                    remove_nodes.append(name)
                    continue
            try:
                i_pos = lat_long[name]
                G_sub.nodes[name]["pos"] = i_pos
            except Exception as e:
                print(e)
                remove_nodes.append(name)

        for name in remove_nodes:
            G_sub.remove_node(name)

        width = [G_sub.get_edge_data(i, j, k)["v"] for i, j, k in G_sub.edges]
        width = np.asarray(width)

        if log_scale:
            width = np.log(width)
        if normalize:
            width = (width - width.min()) / (width.max() - width.min()) + 0.5
        if quantile:
            q = np.quantile(a=width, q=quantile)

        for j, edge in enumerate(G_sub.edges):
            if quantile and (width[j] < q):
                continue
            if G_sub.nodes[0]["attr"]:
                weights = []
                for node in G_sub.nodes:
                    weights.append(G_sub[node]["attr"])
            else:
                weights = 10
            xy1 = G_sub.nodes[edge[0]]["pos"]
            xy2 = G_sub.nodes[edge[1]]["pos"]
            plt.scatter([xy1[0], xy2[0]], [xy1[1], xy2[1]], color="b", s=weights)
            ax.annotate(
                "",
                xy=xy2,
                xycoords="data",
                xytext=xy1,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[i],
                    shrinkA=5,
                    shrinkB=5,
                    patchA=None,
                    patchB=None,
                    connectionstyle=f"arc3,rad={radius[i]}",
                    lw=width[j],
                ),
                transform=ccrs.Geodetic(),
            )

    ax.stock_img()
    ax.add_feature(cfeature.BORDERS, alpha=0.5, linestyle=":")
    ax.add_feature(cfeature.COASTLINE, alpha=0.5)


def relational_graph_plotter(graph):
    if isinstance(graph, tf.sparse.SparseTensor):
        graph = tf.sparse.to_dense(graph)
    fig, ax = plt.subplots(graph.shape[0])
    for i in range(graph.shape[0], 1):
        ax[i, 0].imshow(graph[i, :, :], cmap="winter")
    plt.show()