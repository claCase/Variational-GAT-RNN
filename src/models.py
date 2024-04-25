from tensorflow_probability.python.distributions import (
    MultivariateNormalDiag,
    Bernoulli,
    Independent,
    kl_divergence,
)
import tensorflow as tf
from .layers import (
    GNNRNNCell,
    NestedGRUAttentionCell,
    GATv2Layer,
    VGRNNCell,
    BatchMultiBilinearDecoderDense,
)
from .losses import custom_mse, zero_inflated_nlikelihood
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from .utils import zero_inflated_lognormal

m = tf.keras.models
l = tf.keras.layers
act = tf.keras.activations
init = tf.keras.initializers


def build_RNNGAT(distribution, **kwargs):
    kwargs_ = dict(
        nodes=171,
        dropout=0,
        recurrent_dropout=0,
        attn_heads=1,
        channels=35,
        out_channels=30,
        activation="gelu",
        gnn_type="gat",
        add_bias=True,
        h_gnn=True,
    )
    kwargs_.update(kwargs)
    if distribution != "bernulli":

        @tf.keras.utils.register_keras_serializable(package="GNNRNN")
        def lognormal_loss(labels, logits):
            labels = tf.expand_dims(labels, -1)
            return zero_inflated_nlikelihood(
                labels, logits, mean_axis=None, distribution=distribution
            )

        if distribution == "halfnormal":
            depth = 2
        else:
            depth = 3
        loss = lognormal_loss
    else:
        @tf.keras.utils.register_keras_serializable(package="GNNRNN")
        def bce(labels, logits):
            return tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    labels[..., None], logits[..., None], from_logits=True
                )
            )
        loss = bce
        depth = 1

    i1, i2 = tf.keras.Input(shape=(None, 171, 171)), tf.keras.Input(
        shape=(None, 171, 171)
    )
    rnn_gat = RNNGAT(**kwargs_)
    decoder = BatchMultiBilinearDecoderDense(
        depth=depth, diagonal=False, mask_diag=True, activation="linear"
    )
    xo, xp = rnn_gat((i1, i2))
    xa = decoder(xo)
    model = tf.keras.models.Model((i1, i2), (xa))
    model.compile(optimizer="rmsprop", loss=loss, metrics=loss)
    return model


# logits = model2((x, adj), training=True)
@tf.keras.utils.register_keras_serializable("GNNRNN")
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
        gnn_type="gat",
        h_gnn=True,
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
        self.gnn_type = gnn_type
        self.h_gnn = h_gnn
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.out_channels = out_channels

        rnn_cell = GNNRNNCell(
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
            gnn_type=self.gnn_type,
            h_gnn=self.h_gnn,
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
    def call(self, inputs, states=None, training=None, mask=None):
        x, a = inputs
        return self.rnn((x, a), initial_state=states, training=training)

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
            "out_channels": self.out_channels,
            "concat_heads": self.concat_heads,
            "add_bias": self.add_bias,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "regularizer": self.regularizer,
            "return_attn_coef": self.return_attn_coef,
            "layer_norm": self.layer_norm,
            "initializer": self.initializer,
            "gnn_type": self.gnn_type,
            "h_gnn": self.h_gnn,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "stateful": self.stateful,
            "unroll": self.unroll,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable("GNNRNN")
class VRNNGATBinary(m.Model):
    def __init__(
        self,
        nodes,
        dropout,
        recurrent_dropout,
        attn_heads,
        channels,
        concat_heads=False,
        add_bias=False,
        diagonal=False,
        activation="elu",
        output_activation="tanh",
        regularizer=None,
        return_attn_coef=False,
        layer_norm=False,
        initializer="glorot_normal",
        gat_type="gat",
        single_gat=False,
        return_sequences=True,
        return_state=True,
        go_backwards=False,
        stateful=False,
        unroll=False,
        adj_pos=False,  # positive edges re-weighting
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
        self.diagonal = diagonal
        self.activation = activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.return_attn_coef = return_attn_coef
        self.layer_norm = layer_norm
        self.initializer = initializer
        self.gat_type = gat_type
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.single_gat = single_gat
        self.adj_pos = adj_pos
        self.kl_weight = 1e-4

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
            gat_type=self.gat_type,
            outputs=1,
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
    def likelihood(self, true_adj, pred_adj, weighted=False):
        pos = tf.cast(true_adj > 0, tf.float32)
        if weighted:
            B = tf.cast(tf.shape(true_adj)[0], tf.float32)
            T = tf.cast(true_adj.shape[1], tf.float32)
            pos_sum = tf.reduce_sum(pos)
            tot = T * B * float(self.nodes) ** 2
            neg = tot - pos_sum  # negative edges
            posw = neg / pos_sum
            norm = tot / neg
        else:
            posw = 1.0
            norm = 1.0
        loss = tf.keras.losses.binary_crossentropy(
            y_true=true_adj[..., None], y_pred=pred_adj[..., None], from_logits=True
        )
        # loss = loss * (1 - pos) + loss * pos * posw
        # loss = loss * norm
        return -tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, (-1, -2)), 1))

    @tf.function
    def kl_hidden(self, mu_prior, sigma_prior, mu_posterior, sigma_posterior):
        B = tf.shape(mu_prior)[0]
        T = tf.shape(mu_prior)[1]
        N = tf.shape(mu_prior)[2]
        # Reshaping to independent gaussians -> (B, T, N, d)
        shape = (B, T, N, -1)
        mu_prior = tf.reshape(mu_prior, shape)
        sigma_prior = tf.reshape(sigma_prior, shape)
        mu_posterior = tf.reshape(mu_posterior, shape)
        sigma_posterior = tf.reshape(sigma_posterior, shape)
        distr_prior = MultivariateNormalDiag(mu_prior, sigma_prior)
        distr_posterior = MultivariateNormalDiag(mu_posterior, sigma_posterior)
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(kl_divergence(distr_posterior, distr_prior), -1), 1
            )
        )
    
    @tf.function
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = tf.cast(tf.shape(mean_1)[-2], mean_1.dtype)
        kld_element = (
            2 * tf.math.log(std_2)
            - 2 * tf.math.log(std_1)
            + (tf.math.pow(std_1, 2) + tf.math.pow(mean_1 - mean_2, 2))
            / tf.math.pow(std_2, 2)
            - 1
        )
        return tf.reduce_mean(tf.reduce_sum(kld_element, axis=-1)) * (0.5 / num_nodes)


    @tf.function
    def call(self, inputs, states=None, training=None, mask=None):
        return self.rnn(inputs, training=training)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            *o, h = self(x, training=True)
            if self.return_attn_coef:
                o = o[1:]
            mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o
            adj_dec = tf.squeeze(adj_dec, axis=-1)
            nll = - self.likelihood(y, adj_dec)
            #kl = self.kl_hidden(mu_prior, sigma_prior, post_t_mu, post_t_sigma)
            kl = self._kld_gauss(post_t_mu, post_t_sigma, mu_prior, sigma_prior)
            loss = nll + kl * self.kl_weight
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker_nll.update_state(nll)
        self.loss_tracker_kl.update_state(kl)
        return {
            "nll": self.loss_tracker_nll.result(),
            "kl": self.loss_tracker_kl.result(),
        }

    @tf.function
    def test_step(self, data):
        x, y = data
        *o, h = self(x, training=True)
        if self.return_attn_coef:
            mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o[1:]
        else:
            mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o
        nll = self.likelihood(y, adj_dec, self.adj_pos)
        kl = self._kld_gauss(post_t_mu, post_t_sigma, mu_prior, sigma_prior)
        return {"nll": nll, "kl": kl}

    def sample_adjacency(self, inputs, n_samples=1):
        *_, adj_dec, h = self(inputs, training=False)
        adj_dec = tf.squeeze(adj_dec, -1)
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


@tf.keras.utils.register_keras_serializable("GNNRNN")
class VRNNGATWeighted(m.Model):
    def __init__(
        self,
        nodes,
        dropout,
        recurrent_dropout,
        attn_heads,
        channels,
        distribution="lognormal",
        concat_heads=False,
        add_bias=True,
        diagonal=False,
        activation="relu",
        output_activation="tanh",
        regularizer=None,
        return_attn_coef=False,
        layer_norm=False,
        initializer="glorot_normal",
        gat_type="gat",
        return_sequences=True,
        return_state=True,
        go_backwards=False,
        stateful=False,
        unroll=False,
        adj_pos=False,
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
        self.diagonal = diagonal
        self.activation = activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.return_attn_coef = return_attn_coef
        self.layer_norm = layer_norm
        self.initializer = initializer
        self.gat_type = gat_type
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.distribution = distribution
        self.n_outputs = 2 if distribution == "halfnormal" else 3
        self.adj_pos = adj_pos
        self.kl_weight = 1e-4

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
            gat_type=self.gat_type,
            outputs=self.n_outputs,
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
    def likelihood(self, true_adj, pred_adj, weighted=False):
        """Computes Likelihood of temporal batched adjeciency matrix

        Args:
            true_adj (tf.Tensor): (B, T, N, N)
            pred_adj (tf.Tensor): (B, T, N, N)

        Returns:
            tf.Tensor: (B,T)
        """
        if weighted:
            B = tf.cast(tf.shape(true_adj)[0], tf.float32)
            T = tf.cast(true_adj.shape[1], tf.float32)
            pos = tf.cast(true_adj > 0, tf.float32)
            pos_sum = tf.reduce_sum(pos)
            tot = T * B * float(self.nodes) ** 2
            neg = tot - pos_sum  # negative edges
            posw = neg / pos_sum
            norm = tot / neg
        else:
            posw = 1.0
            norm = 1.0
        true_adj = tf.expand_dims(true_adj, -1)
        loss = zero_inflated_nlikelihood(
            labels=true_adj, logits=pred_adj, mean_axis=None, pos_weight=posw
        )
        loss = norm * loss
        return -loss

    @tf.function
    def kl_hidden(self, mu_prior, sigma_prior, mu_posterior, sigma_posterior):
        B = tf.shape(mu_prior)[0]
        T = tf.shape(mu_prior)[1]
        N = tf.shape(mu_prior)[2]
        # Reshaping to independent gaussians -> (B, T, N, d)
        shape = (B, T, N, -1)
        mu_prior = tf.reshape(mu_prior, shape)
        sigma_prior = tf.reshape(sigma_prior, shape)
        mu_posterior = tf.reshape(mu_posterior, shape)
        sigma_posterior = tf.reshape(sigma_posterior, shape)
        distr_prior = MultivariateNormalDiag(mu_prior, sigma_prior)
        distr_posterior = MultivariateNormalDiag(mu_posterior, sigma_posterior)
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(kl_divergence(distr_posterior, distr_prior), -1), 1
            )
        )

    @tf.function
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = tf.cast(tf.shape(mean_1)[-2], mean_1.dtype)
        kld_element = (
            2 * tf.math.log(std_2)
            - 2 * tf.math.log(std_1)
            + (tf.math.pow(std_1, 2) + tf.math.pow(mean_1 - mean_2, 2))
            / tf.math.pow(std_2, 2)
            - 1
        )
        return tf.reduce_mean(tf.reduce_sum(kld_element, axis=-1)) * (0.5 / num_nodes)

    @tf.function
    def call(self, inputs, states=None, training=None, mask=None):
        return self.rnn(inputs, training=training)

    @tf.function
    def sample(self, inputs=None, logits=None, samples=1):
        if inputs is not None:
            *_, logits, _ = self(inputs)
        elif logits is None:
            raise ValueError("If inputs is None logits must not be None")
        return zero_inflated_lognormal(logits).sample(samples)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            *o, h = self(x, training=True)
            if self.return_attn_coef:
                o = o[1:]
            mu_prior, sigma_prior, post_t_mu, post_t_sigma, h_prime, adj_dec = o
            nll = -self.likelihood(y, adj_dec, self.adj_pos)
            kl = self.kl_hidden(mu_prior, sigma_prior, post_t_mu, post_t_sigma)
            loss = nll + kl * self.kl_weight
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

    """def sample_adjacency(self, n_samples=1):
        *o, h = self(x, training=False)
        adj_dec = o[-1]
        adj_distr = Independent(Bernoulli(logits=adj_dec), reinterpreted_batch_ndims=2)
        return adj_distr.sample(n_samples)"""

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
