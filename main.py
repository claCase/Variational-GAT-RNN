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
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--model",type=str, default="gatrnn")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--plot_network", action="store_true")
    args = parser.parse_args()
    start = args.start 
    end = args.end 
    t = end-start // 2
    prod = args.prod
    epochs = args.epochs 
    model = args.model 
    binary = args.binary 
    plot_network = args.plot_network 

    baci_loader = data.BaciDataLoader()
    try:
        baci_loader.load()
    except:
        try:
            for i in range(start, end, 2):
                baci_loader2 = data.BaciDataLoader()
                baci_loader2.from_csv_path(path, i, i + 1)
                baci_loader2.remove_countries()
                baci_loader2.reduce_product_detail(2)
                baci_loader2.select_products([prod])
                baci_loader2.fill_missing_with_avg()
                baci_loader.join(baci_loader2)
                del baci_loader2
        except Exception as e:
            raise e
    
    N = len(baci_loader.countries())
    years = baci_loader.years
    spt = baci_loader.to_tf_sparse()
    spt = tf.sparse.reorder(spt)
    dense = tf.sparse.to_dense(tf.sparse.slice(spt, (0,0,0,0,0), (years, N, N, 1, 1)))
    adj = tf.cast(tf.expand_dims(tf.squeeze(dense, -1), 0), tf.float32)
    adj_in = tf.where(adj>0., tf.math.log(adj), adj)
    x = tf.cast(tf.constant([np.eye(N)]*years)[None,:], tf.float32)

    if binary:
        distribution = "bernulli"
    else:
        distribution = "lognromal"

    if model == "gatrnn":
        model = models.build_RNNGAT(distribution)    
        model.save(f"./saved_models/{distribution}_gatrnn.keras")
        logits, _  = model((x[:, :-1], adj[:,:-1, :, :, 0]))

    elif model == "vgrnn":
        kl_schedule = utils.Scheduler("kl_weight", 100)
        if distribution=="binary":
            model = models.VRNNGATBinary(nodes=N, dropout=0.01, recurrent_dropout=0.01, attn_heads=1, channels=15)
        else:
            model = models.VRNNGATWeighted(nodes=N, dropout=0.01, recurrent_dropout=0.01, attn_heads=1, channels=15)
        model.compile("adam")
        _ = model((x[:, :-1], adj[:,:-1, :, :, 0]))
        model.fit((x[:, :-1], adj[:,:-1, :, :, 0]), adj[:,1:, :, :,0], epochs=epochs, callbacks=[kl_schedule])
        model.save(f"./saved_models/{distribution}_vgatrnn.keras")
        *_, logits, _  = model((x[:, :-1], adj[:,:-1, :, :, 0]))

    if plot_network: 
        if distribution=="lognormal":
            plotter.plot_dynamic_adj(logits[0])
        else:
            plotter.plot_dynamic_bernulli_adj(logits[0])
        G = baci_loader.to_networkx()
        fig = go.Figure(layout=dict(width=1000, height=2500))
        plotter.draw_subgraph(G, ['84', '08', '57'], bottom_quantile=.98, top_quantile=1., log_scale=True, normalize=True, plotly=True, fig=fig)
        fig.show()

        fig = plt.figure(figsize=(15, 15))
        plotter.draw_subgraph(G, ['84', '08', '57'], bottom_quantile=.98, top_quantile=1., log_scale=True, normalize=True, plotly=False, fig=fig)
        fig.show()