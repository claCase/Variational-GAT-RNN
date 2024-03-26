from typing import List
import tensorflow as tf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from src.utils import (
    get_product_subgraph,
    country_code_converter,
    country_long_lat,
    slerp,
)
import os
import pandas as pd


def draw_subgraph(
    G: nx.MultiDiGraph,
    prod_keys: List[str],
    nodes: List[str] = None,
    value_key: str = "v",
    log_scale=True,
    normalize=True,
    bottom_quantile=0.9,
    top_quantile=1,
    attribute_key=None,
    plotly=False,
    use_mapbox=False,
    fig=None,
    ax=None,
):
    assert bottom_quantile < top_quantile

    radius = {
        prod: value
        for prod, value in zip(prod_keys, np.linspace(-0.1, -0.5, len(prod_keys)))
    }
    color_idx = {prod: idx for idx, prod in enumerate(prod_keys)}

    G_sub = get_product_subgraph(G, prod_keys=prod_keys)
    remove_nodes = []
    for name in G_sub.nodes:
        if nodes is not None:
            if name not in nodes:
                remove_nodes.append(name)
                continue
        try:
            i_pos = None
            if type(name) is int:
                iso2 = country_code_converter([name], "iso2")[0]
                iso3 = country_code_converter([name], "iso3")[0]
                i_pos = country_long_lat([iso2])[0]
                G_sub.nodes[name]["iso2"] = iso2
                G_sub.nodes[name]["iso3"] = iso3
            else:
                # print(f"name: {name}")
                i_pos = country_long_lat([iso2])[0]
            if i_pos is None:
                remove_nodes.append(name)
            else:
                G_sub.nodes[name]["pos"] = i_pos
        except Exception as e:
            print(e)
            remove_nodes.append(name)

    for name in remove_nodes:
        G_sub.remove_node(name)

    top_prod_quantiles = {prod: [] for prod in prod_keys}
    bot_prod_quantiles = {prod: [] for prod in prod_keys}
    min_prod = {prod: 0 for prod in prod_keys}
    for key in prod_keys:
        width = np.asarray(
            [G_sub.get_edge_data(i, j, k)["v"] for i, j, k in G_sub.edges if k == key]
        )
        if log_scale:
            min_width = width.min()
            width = np.log(width + min_width)
            min_prod[key] = min_width

        b_q = np.quantile(a=width, q=bottom_quantile)
        bot_prod_quantiles[key] = b_q
        t_q = np.quantile(a=width, q=top_quantile)
        top_prod_quantiles[key] = t_q

    if attribute_key is not None:
        try:
            attr = nx.get_node_attributes(G_sub, attribute_key)
            weights = list(attr.values())
        except Exception as e:
            raise e
    else:
        weights = 10

    long_lat = np.asarray(list(nx.get_node_attributes(G_sub, "pos").values()))

    colors = plt.get_cmap("Set1")(range(len(prod_keys)))

    # plt.scatter(long_lat[:, 0], long_lat[:, 1], color="b", s=weights)

    if not plotly:
        if fig is None:
            fig = plt.figure(figsize=(25, 25))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        for j, edge in enumerate(G_sub.edges):
            prod = edge[2]
            width = G_sub.edges[edge]["v"]

            if log_scale:
                width = np.log(width + min_prod[prod])

            if not (bot_prod_quantiles[prod] < width < top_prod_quantiles[prod]):
                continue

            normed_width = (width - bot_prod_quantiles[prod]) / (
                top_prod_quantiles[prod] - bot_prod_quantiles[prod]
            )

            if normalize:
                width = normed_width

            e1 = G_sub.nodes[edge[0]]
            xy1 = e1["pos"]
            e2 = G_sub.nodes[edge[1]]
            xy2 = e2["pos"]
            ax.scatter(
                xy1[0],
                xy1[1],
                s=weights[e1] if type(weights) is not int else weights,
                color="b",
            )
            ax.scatter(
                xy2[0],
                xy2[1],
                s=weights[e2] if type(weights) is not int else weights,
                color="b",
            )
            ax.annotate(
                "",
                xy=xy2,
                xycoords="data",
                xytext=xy1,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[color_idx[prod]],
                    alpha=normed_width,
                    shrinkA=5,
                    shrinkB=5,
                    patchA=None,
                    patchB=None,
                    connectionstyle=f"arc3,rad={radius[prod]}",
                    lw=width * 2,
                ),
                transform=ccrs.Geodetic(),
            )
        ax.stock_img()
        ax.add_feature(cfeature.BORDERS, alpha=0.5, linestyle=":")
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        return fig, ax
    else:
        df_nodes = pd.DataFrame()
        nodes = list(G_sub.nodes)
        df_nodes["code"] = nodes
        df_nodes["name"] = country_code_converter(nodes, "name")
        df_nodes["iso3"] = [G_sub.nodes[node]["iso3"] for node in nodes]
        df_nodes["long"] = [G_sub.nodes[node]["pos"][0] for node in nodes]
        df_nodes["lat"] = [G_sub.nodes[node]["pos"][1] for node in nodes]
        draw_nodes = set()
        fig = go.Figure()

        for j, edge in enumerate(G_sub.edges):
            prod = edge[2]
            width = G_sub.edges[edge]["v"]

            if log_scale:
                width = np.log(width + min_prod[prod])

            if not (bot_prod_quantiles[prod] < width < top_prod_quantiles[prod]):
                continue

            normed_width = (width - bot_prod_quantiles[prod]) / (
                top_prod_quantiles[prod] - bot_prod_quantiles[prod]
            )
            if normalize:
                width = normed_width

            e1 = G_sub.nodes[edge[0]]
            xy1 = e1["pos"]
            e2 = G_sub.nodes[edge[1]]
            xy2 = e2["pos"]
            draw_nodes.update([edge[0]])
            draw_nodes.update([edge[1]])
            if use_mapbox:
                lons, lats = slerp(xy1, xy2, dir=1)
                fig.add_trace(
                    go.Scattermapbox(
                        lon=lons,
                        lat=lats,
                        mode="lines",
                        line=dict(
                            width=1,
                            color=px.colors.qualitative.Light24[color_idx[prod]],
                        ),
                        opacity=normed_width,
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergeo(
                        locationmode="USA-states",
                        lon=[xy1[0], xy2[0]],
                        lat=[xy1[1], xy2[1]],
                        mode="lines",
                        line=dict(
                            width=1,
                            color=px.colors.qualitative.Light24[color_idx[prod]],
                        ),
                        opacity=normed_width,
                    )
                )

        if not use_mapbox:
            draw_df_nodes = df_nodes[df_nodes["code"].isin(draw_nodes)]
            # print(draw_df_nodes, draw_nodes)
            fig.add_trace(
                go.Scattergeo(
                    locationmode="ISO-3",
                    lon=draw_df_nodes["long"],
                    lat=draw_df_nodes["lat"],
                    hoverinfo="text",
                    text=draw_df_nodes["name"],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color="Blue",
                        # line=dict(width=3, color="rgba(68, 68, 68, 0)"),
                    ),
                )
            )
            fig.update_layout(
                width=1000, 
                height=1000,
                autosize=True,
                # title_text=f"Trade for products {prod_keys} for countries between {bottom_quantile} and {top_quantile} quantiles",
                showlegend=False,
                geo=dict(
                    scope="world",
                    # projection_type="azimuthal equal area",
                    projection_type="natural earth",                    
                ),
            )
            fig.update_geos(
                resolution=50,
                showcoastlines=True,
                coastlinecolor="RebeccaPurple",
                showland=True,
                landcolor="LightGreen",
                showocean=True,
                oceancolor="LightBlue",
                showcountries=True,
                countrycolor="RebeccaPurple"
                
            )
        else:
            fig.update_layout(
                    showlegend=False,
                    mapbox_style="open-street-map",
                    mapbox_zoom=1,
                    mapbox_center_lat=41,
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                )
    return fig 

def relational_graph_plotter(graph):
    if isinstance(graph, tf.sparse.SparseTensor):
        graph = tf.sparse.to_dense(graph)
    fig, ax = plt.subplots(graph.shape[0])
    for i in range(graph.shape[0], 1):
        ax[i, 0].imshow(graph[i, :, :], cmap="winter")
    plt.show()
