from src import data, models, layers, utils, losses, plotter
from importlib import reload 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import networkx as nx
import plotly.graph_objects as go 
import os 
import argparse

path = os.environ["DATA"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1995)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--prod", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    start = args.start 
    end = args.end 
    t = end-start // 2
    prod = args.prod
    epochs = args.epochs 

    baci_loader = data.BaciDataLoader()
    for i in range(0, t, 2):
        baci_loader2 = data.BaciDataLoader()
        baci_loader2.from_csv_path(path, start+i, start + 1 +i)
        baci_loader2.remove_countries()
        baci_loader2.fill_missing_with_avg()
        baci_loader2.reduce_product_detail(2)
        baci_loader.join(baci_loader2)
        del baci_loader2
    
    spt = baci_loader.to_tf_sparse()
    spt = tf.sparse.reorder(spt)
    dense = tf.sparse.to_dense(tf.sparse.slice(spt, (0,0,0,prod + 1,0), (8, 170, 170, prod + 1, 1)))
    adj = tf.cast(tf.expand_dims(tf.squeeze(dense, -1), 0), tf.float32)
    adj_in = tf.where(adj>0., tf.math.log(adj), adj)
    x = tf.cast(tf.constant([np.eye(170)]*8)[None,:], tf.float32)


    model = models.VRNNGATWeighted(nodes=170, dropout=0.01, recurrent_dropout=0.01, attn_heads=4, channels=10)
    model.compile("adam")
    _ = model((x[:, :-1], adj[:,:-1, :, :, 0]))
    model.fit((x[:, :-1], adj_in[:,:-1, :, :, 0]), adj[:,1:, :, :,0], epochs=epochs)

    fig = go.Figure(layout=dict(width=1000, height=2500))
    plotter.draw_subgraph(G, ['84', '08', '57'], bottom_quantile=.98, top_quantile=1., log_scale=True, normalize=True, plotly=True, fig=fig)
    fig.show()

    fig = plt.figure(figsize=(15, 15))
    plotter.draw_subgraph(G, ['84', '08', '57'], bottom_quantile=.98, top_quantile=1., log_scale=True, normalize=True, plotly=False, fig=fig)
    fig.show()