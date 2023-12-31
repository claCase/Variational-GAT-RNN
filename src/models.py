from tensorflow_probability.python.distributions import (
    MultivariateNormalDiag,
    Bernoulli,
    Independent,
    kl_divergence,
)
import tensorflow as tf
from .layers import (
    NestedGRUGATCell,
    NestedGRUAttentionCell,
    NestedGRUGATCellSingle,
    GATv2Layer,
    BatchBilinearDecoderDense,
    VGRNNCell,
)
from .losses import custom_mse
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.distributions import MultivariateNormalDiag

tf.keras.utils.get_custom_objects().clear()

m = tf.keras.models
l = tf.keras.layers
act = tf.keras.activations
init = tf.keras.initializers


def build_RNNGAT(
    nodes,
    input_features,
    dropout=0.01,
    activation="relu",
    recurrent_dropout=0.01,
    regularizer=None,
    layer_norm=False,
    gatv2=True,
    concat_heads=False,
    return_attn_coef=False,
    attn_heads=4,
    channels=15,
    stateful=True,
    return_sequences=True,
    return_state=True,
    add_bias=True,
    units=1,
    output_activation="tanh",
    single=True,
    initializer="glorot_normal",
):
    i1 = tf.keras.Input(shape=(None, nodes, input_features), batch_size=1)
    i2 = tf.keras.Input(shape=(None, nodes, nodes), batch_size=1)
    if single:
        cell = NestedGRUGATCellSingle(
            nodes,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            attn_heads=attn_heads,
            channels=channels,
            concat_heads=concat_heads,
            add_bias=add_bias,
            activation=activation,
            regularizer=regularizer,
            return_attn_coef=return_attn_coef,
            layer_norm=layer_norm,
            initializer=initializer,
            gatv2=gatv2,
        )
    else:
        cell = NestedGRUGATCell(
            nodes,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            attn_heads=attn_heads,
            channels=channels,
            concat_heads=concat_heads,
            add_bias=add_bias,
            activation=activation,
            regularizer=regularizer,
            return_attn_coef=return_attn_coef,
            layer_norm=layer_norm,
            initializer=initializer,
            gatv2=gatv2,
        )

    rnn = l.RNN(
        cell,
        stateful=stateful,
        return_sequences=return_sequences,
        return_state=return_state,
        time_major=False,
    )
    pred_layer = l.Dense(units=units, activation=output_activation)
    o, h = rnn((i1, i2))
    if return_attn_coef:
        p = pred_layer(o[0])
    else:
        p = pred_layer(o)
    model = tf.keras.models.Model([i1, i2], [o, p])
    return model


def build_RNNAttention(
    nodes,
    input_features,
    kwargs_cell={
        "dropout": 0.01,
        "activation": "relu",
        "recurrent_dropout": 0.01,
        "hidden_size_out": 15,
        "regularizer": None,
        "layer_norm": False,
    },
    kwargs_rnn={"stateful": False, "return_sequences": True, "return_state": True},
    kwargs_out={"activation": "tanh"},
):
    i1 = tf.keras.Input(shape=(None, nodes, input_features), batch_size=None)
    i2 = tf.keras.Input(shape=(None, nodes, nodes), batch_size=None)
    cell = NestedGRUAttentionCell(nodes, **kwargs_cell)
    rnn = l.RNN(cell, **kwargs_rnn, time_major=False)
    pred_layer = l.Dense(1, **kwargs_out)
    o, h = rnn((i1, i2))
    p = pred_layer(o)
    return tf.keras.models.Model([i1, i2], [o, p])


class RNNGAT(m.Model):
    def __init__(
        self,
        nodes,
        dropout,
        recurrent_dropout,
        attn_heads,
        channels,
        out_channels,
        concat_heads=False,
        add_bias=True,
        activation="relu",
        output_activation="tanh",
        regularizer=None,
        return_attn_coef=False,
        layer_norm=False,
        initializer="glorot_normal",
        gatv2=True,
        single_gnn=True,
        return_sequences=True,
        return_state=True,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.nodes = nodes
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.attn_heads = attn_heads
        self.channels = channels
        self.concat_heads = concat_heads
        self.add_bias = add_bias
        self.activation = activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.return_attn_coef = return_attn_coef
        self.layer_norm = layer_norm
        self.initializer = initializer
        self.gatv2 = gatv2
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.single_gnn = single_gnn
        self.out_channels = out_channels

        if self.single_gnn:
            rnn_cell = NestedGRUGATCellSingle(
                nodes=self.nodes,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                attn_heads=self.attn_heads,
                channels=self.channels,
                concat_heads=self.concat_heads,
                add_bias=self.add_bias,
                activation=self.activation,
                regularizer=self.regularizer,
                return_attn_coef=self.return_attn_coef,
                layer_norm=self.layer_norm,
                initializer=self.initializer,
                gatv2=self.gatv2,
            )
        else:
            rnn_cell = NestedGRUGATCell(
                nodes=self.nodes,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                attn_heads=self.attn_heads,
                channels=self.channels,
                concat_heads=self.concat_heads,
                add_bias=self.add_bias,
                activation=self.activation,
                regularizer=self.regularizer,
                return_attn_coef=self.return_attn_coef,
                layer_norm=self.layer_norm,
                initializer=self.initializer,
                gatv2=self.gatv2,
            )

        self.rnn = l.RNN(
            rnn_cell,
            return_sequences=self.return_sequences,
            return_state=self.return_state,
            go_backwards=self.go_backwards,
            stateful=self.stateful,
            unroll=self.unroll,
            time_major=False,
        )
        self.pred_out = l.Dense(self.out_channels, self.output_activation)

    def build(self, input_shape):
        x, a = input_shape
        self.rnn.build((x, a))
        rnn_out_shape = self.rnn.compute_output_shape((x, a))
        self.pred_out.build(rnn_out_shape[0])

    @tf.function
    def call(self, inputs, states=None, training=None, mask=None):
        x, a = inputs
        o, h = self.rnn((x, a), initial_state=states, training=training)
        if self.return_attn_coef:
            p = self.pred_out(o[0])
        else:
            p = self.pred_out(o)
        return o, p

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            o, y_pred = self(x, training=True)
            l = self.compute_loss(y=y, y_pred=y_pred)
        grads = tape.gradient(l, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(l)
        return {"Training Loss": self.loss_tracker.result()}

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        return self.compiled_loss(y, y_pred)

    def test_step(self, data):
        x, y = data
        o, y_pred = self(x)
        loss = self.compiled_loss(y, y_pred)
        return {"Test Loss": loss}

    def get_config(self):
        config = {
            "nodes": self.nodes,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "attn_heads": self.attn_heads,
            "channels": self.channels,
            "concat_heads": self.concat_heads,
            "add_bias": self.add_bias,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "regularizer": self.regularizer,
            "return_attn_coef": self.return_attn_coef,
            "layer_norm": self.layer_norm,
            "initializer": self.initializer,
            "gatv2": self.gatv2,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "stateful": self.stateful,
            "unroll": self.unroll,
            "time_major": self.time_major,
            "single_gnn": self.single_gnn,
            "out_channels": self.out_channels,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config, custom_objects=None):
        if custom_objects.get("loss"):
            cls.compiled_loss = custom_objects["loss"]
        return cls(**config)


class VRNNGAT(m.Model):
    def __init__(
        self,
        nodes,
        dropout,
        recurrent_dropout,
        attn_heads,
        channels,
        concat_heads=False,
        add_bias=True,
        activation="relu",
        output_activation="tanh",
        regularizer=None,
        return_attn_coef=False,
        layer_norm=False,
        initializer="glorot_normal",
        gatv2=True,
        return_sequences=True,
        return_state=True,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_tracker_nll = tf.keras.metrics.Mean(name="nll")
        self.loss_tracker_kl = tf.keras.metrics.Mean(name="kl")

        self.nodes = nodes
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.attn_heads = attn_heads
        self.channels = channels
        self.concat_heads = concat_heads
        self.add_bias = add_bias
        self.activation = activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.return_attn_coef = return_attn_coef
        self.layer_norm = layer_norm
        self.initializer = initializer
        self.gatv2 = gatv2
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.gnn_post_mu = GATv2Layer(self.attn_heads, self.channels)
        self.gnn_post_sigma = GATv2Layer(self.attn_heads, self.channels)

        rnn_cell = VGRNNCell(
            nodes=self.nodes,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            attn_heads=self.attn_heads,
            channels=self.channels,
            concat_heads=self.concat_heads,
            add_bias=self.add_bias,
            activation=self.activation,
            regularizer=self.regularizer,
            return_attn_coef=self.return_attn_coef,
            layer_norm=self.layer_norm,
            initializer=self.initializer,
            gatv2=self.gatv2,
        )

        self.rnn = l.RNN(
            rnn_cell,
            return_sequences=self.return_sequences,
            return_state=self.return_state,
            go_backwards=self.go_backwards,
            stateful=self.stateful,
            unroll=self.unroll,
            time_major=False,
        )

    @tf.function
    def likelihood(self, true_adj, pred_adj):
        adj_distr = Independent(Bernoulli(logits=pred_adj), reinterpreted_batch_ndims=2)
        return tf.reduce_sum(adj_distr.log_cdf(true_adj), 1)

    @tf.function
    def kl_hidden(self, mu_prior, sigma_prior, mu_posterior, sigma_posterior):
        distr_prior = MultivariateNormalDiag(mu_prior, sigma_prior)
        distr_posterior = MultivariateNormalDiag(mu_posterior, sigma_posterior)
        return tf.reduce_sum(kl_divergence(distr_posterior, distr_prior), 1)

    @tf.function
    def call(self, inputs, states=None, training=None, mask=None):
        return self.rnn(inputs)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            *o, h = self(x, training=True)
            if self.return_attn_coef:
                mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o[1:]
            else:
                mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o
            nll = -self.likelihood(y, adj_dec)
            kl = -self.kl_hidden(mu_prior, sigma_prior, post_t_mu, post_t_sigma)
            loss = tf.reduce_mean(nll) + tf.reduce_mean(kl)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker_nll.update_state(nll)
        self.loss_tracker_kl.update_state(kl)
        return {
            "nll": self.loss_tracker_nll.result(),
            "kl": self.loss_tracker_kl.result(),
        }

    def test_step(self, data):
        x, y = data
        *o, h = self(x, training=True)
        if self.return_attn_coef:
            mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o[1:]
        else:
            mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o
        nll = self.likelihood(y, adj_dec)
        kl = self.kl_hidden(mu_prior, sigma_prior, post_t_mu, post_t_sigma)
        return {"nll": nll, "kl": kl}

    def sample_adjacency(self, n_samples=1):
        *o, h = self(x, training=False)
        adj_dec = o[-1]
        adj_distr = Independent(Bernoulli(logits=adj_dec), reinterpreted_batch_ndims=2)
        return adj_distr.sample(n_samples)

    def get_config(self):
        config = {
            "nodes": self.nodes,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "attn_heads": self.attn_heads,
            "channels": self.channels,
            "concat_heads": self.concat_heads,
            "add_bias": self.add_bias,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "regularizer": self.regularizer,
            "return_attn_coef": self.return_attn_coef,
            "layer_norm": self.layer_norm,
            "initializer": self.initializer,
            "gatv2": self.gatv2,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "stateful": self.stateful,
            "unroll": self.unroll,
            "time_major": self.time_major,
            "single_gnn": self.single_gnn,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config, custom_objects=None):
        if custom_objects.get("loss"):
            cls.compiled_loss = custom_objects["loss"]
        return cls(**config)


class GATv2Model(m.Model):
    def __init__(
        self,
        hidden_sizes=[10, 10, 7],
        channels=[10, 10, 7],
        heads=[10, 10, 1],
        dropout=0.2,
        activation="elu",
        *args,
        **kwargs
    ):
        super(GATv2Model, self).__init__(**kwargs)
        self.channels = channels
        self.hidden_sizes = hidden_sizes
        self.heads = heads
        self.dropout = l.Dropout(dropout)
        assert len(channels) == len(heads) == len(hidden_sizes)
        self.n_layers = len(hidden_sizes)
        self.activation = activation
        self.gat_layers = [
            GATv2Layer(heads=hh, channels=c, activation=activation, *args)
            for c, hh in zip(self.channels, self.heads)
        ]
        self.dense_layers = [l.Dense(h, activation) for h in self.hidden_sizes]

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        for d, g in zip(self.dense_layers, self.gat_layers):
            x = d(x)
            x = self.dropout(x)
            x = g([x, a])
        return x


if __name__ == "__main__":
    import numpy as np

    B = 1
    N = 100
    f = 14
    T = 50
    x = np.random.normal(size=(B, T, N, f))
    a = np.random.normal(size=(B, T, N, N))

    @tf.function
    def f():
        tf.compat.v1.disable_eager_execution()
        model = RNNGAT(
            nodes=N,
            dropout=0.1,
            recurrent_dropout=0.1,
            attn_heads=4,
            channels=6,
            out_channels=1,
            concat_heads=False,
            add_bias=True,
            activation="relu",
            output_activation="tanh",
            regularizer=None,
            return_attn_coef=True,
            layer_norm=False,
            initializer="glorot_normal",
            gatv2=True,
            single_gnn=True,
            return_sequences=True,
            return_state=True,
            go_backwards=False,
            stateful=False,
            unroll=False,
            time_major=False,
            mincut=5,
        )

        o, p = model([x, a], training=True)
        model.compile(loss=custom_mse)
        model.fit([x[:, :10], a[:, :10]], x[:, :10, :, 0])

    f()
