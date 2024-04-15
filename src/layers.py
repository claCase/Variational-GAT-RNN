import tensorflow_probability.python.distributions as tfd
from tensorflow_probability.python.internal import tensor_util, parameter_properties
import tensorflow as tf
from tensorflow.python.keras.layers.recurrent import (
    DropoutRNNCellMixin,
    _config_for_enable_caching_device,
    _caching_device,
)
from spektral.layers import GATConv, GCNConv
from spektral.layers.ops import unsorted_segment_softmax
from tensorflow_probability.python.distributions import (
    MultivariateNormalDiag,
)
from src.utils import positive_variance

tf.keras.utils.get_custom_objects().clear()

l = tf.keras.layers
act = tf.keras.activations
init = tf.keras.initializers
regu = tf.keras.regularizers


@tf.keras.utils.register_keras_serializable("NestedRNN")
class VGRNNCell(l.Layer):
    def __init__(
        self,
        nodes,
        outputs=1,
        dropout=0.1,
        recurrent_dropout=0.1,
        attn_heads=4,
        channels=10,
        concat_heads=False,
        add_bias=True,
        diagonal=False,
        activation="elu",
        regularizer=None,
        return_attn_coef=False,
        layer_norm=False,
        initializer="glorot_normal",
        gat_type="gatv2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rnn = GNNRNNCell(
            nodes=nodes,
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
            gnn_type=gat_type,
        )
        self.phi_prior_mu = tf.keras.Sequential(
            [
                l.Dense(channels, activation),
                l.Dense(channels, activation),
                l.Dense(channels, "linear"),
            ]
        )
        self.phi_prior_sigma = tf.keras.Sequential(
            [
                l.Dense(channels, activation),
                l.Dense(channels, activation),
                l.Dense(channels, "linear"),
            ]
        )
        self.phi_x = tf.keras.Sequential(
            [
                l.Dense(channels, activation),
                l.Dense(channels, activation),
                l.Dense(channels, activation),
            ]
        )
        self.phi_z = tf.keras.Sequential(
            [
                l.Dense(channels, activation),
                l.Dense(channels, activation),
                l.Dense(channels, activation),
            ]
        )
        if gat_type == "gatv2":
            gat = GATv2Layer
            gat_kwargs = dict(
                attn_heads=attn_heads, channels=channels, use_bias=add_bias
            )
        elif gat_type == "gat":
            gat = GATConv
            gat_kwargs = dict(
                attn_heads=attn_heads, channels=channels, use_bias=add_bias
            )
        elif gat_type == "gcn":
            gat = GCNConv
            gat_kwargs = dict(channels=channels, use_bias=add_bias)
        else:
            raise NotImplementedError(
                f"Gat type {gat_type} not implemented, choose between gat, gatv2, gcn"
            )

        self.enc = gat(**gat_kwargs, activation=activation)
        self.mu_enc = gat(**gat_kwargs, activation="linear")
        self.sigma_enc = gat(**gat_kwargs, activation="linear")
        self.decoder = BatchMultiBilinearDecoderDense(
            activation="linear", diagonal=diagonal, depth=outputs
        )
        self.state_size = tf.TensorShape((nodes, channels))
        self.return_attn_coef = return_attn_coef

        if return_attn_coef:
            self.output_size = [
                tf.TensorShape((attn_heads, nodes, nodes)),  # attention heads
                tf.TensorShape((nodes, channels)),  # prior mu
                tf.TensorShape((nodes, channels)),  # prior sigma
                tf.TensorShape((nodes, channels)),  # posterior mu
                tf.TensorShape((nodes, channels)),  # posterior sigma
                tf.TensorShape((nodes, channels)),  # hidden state
                tf.TensorShape((nodes, nodes, outputs)),  # adj dec
            ]
        else:
            self.output_size = [
                tf.TensorShape((nodes, channels)),  # prior mu
                tf.TensorShape((nodes, channels)),  # prior sigma
                tf.TensorShape((nodes, channels)),  # posterior mu
                tf.TensorShape((nodes, channels)),  # posterior sigma
                tf.TensorShape((nodes, channels)),  # hidden state
                tf.TensorShape((nodes, nodes, outputs)),  # adj dec
            ]

    def call(self, inputs, states, training, **kwargs):
        x, a = inputs
        h = states
        mu_prior, sigma_prior = self.phi_prior_mu(h[0]), self.phi_prior_sigma(h[0])
        sigma_prior = positive_variance(sigma_prior)

        phi_x_t = self.phi_x(x)
        enc_t = self.enc((tf.concat([phi_x_t, h[0]], axis=-1), a))
        post_t_mu = self.mu_enc([enc_t, a])
        post_t_ss = self.sigma_enc([enc_t, a])
        post_t_ss = positive_variance(post_t_ss)

        z_distr_post = MultivariateNormalDiag(post_t_mu, post_t_ss)
        z_sample = tf.squeeze(z_distr_post.sample(1), 0)
        phi_z_t = self.phi_z(z_sample)
        adj_dec = self.decoder(z_sample)

        o, h_prime = self.rnn(
            (tf.concat([phi_x_t, phi_z_t], axis=-1), a), h, training=training
        )
        if self.return_attn_coef:
            return [
                o[0],
                mu_prior,
                sigma_prior,
                post_t_mu,
                post_t_ss,
                h_prime,
                adj_dec,
            ], h_prime
        else:
            return [
                mu_prior,
                sigma_prior,
                post_t_mu,
                post_t_ss,
                h_prime,
                adj_dec,
            ], h_prime


@tf.keras.utils.register_keras_serializable("GNNRNN", "GNNRNNCell")
class GNNRNNCell(DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer):
    def __init__(
        self,
        nodes,
        dropout,
        recurrent_dropout,
        attn_heads,
        channels,
        concat_heads=False,
        add_bias=False,
        activation="relu",
        regularizer=None,
        return_attn_coef=False,
        layer_norm=False,
        initializer=init.glorot_normal,
        gnn_type="gat",
        h_gnn=True,
        **kwargs,
    ):
        super(GNNRNNCell, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.attn_heads = attn_heads
        self.channels = channels
        self.concat_heads = concat_heads
        self.add_bias = add_bias
        self.activation = activation
        self.regularizer = regularizer
        self.return_attn_coef = return_attn_coef
        self.layer_norm = layer_norm
        self.initializer = initializer
        self.gnn_type = gnn_type
        self.h_gnn = h_gnn 
        self.state_size = tf.TensorShape((self.tot_nodes, self.channels))
        if return_attn_coef:
            if self.h_gnn:
                self.output_size = [
                    tf.TensorShape((self.tot_nodes, self.channels)),
                    tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                    tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                    tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                    tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                ]
            else:
                self.output_size = [
                    tf.TensorShape((self.tot_nodes, self.channels)),
                    tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                    tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                ]
        else:
            self.output_size = tf.TensorShape((self.tot_nodes, self.channels))

        if self.layer_norm:
            self.ln = l.LayerNormalization()
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

        if gnn_type == "gatv2":
            gnn = GATv2Layer
            gat_kwargs = dict(
                attn_heads=attn_heads, channels=channels, use_bias=add_bias
            )
        elif gnn_type == "gat":
            gnn = GATConv
            gat_kwargs = dict(
                attn_heads=attn_heads, channels=channels, use_bias=add_bias
            )
        elif gnn_type == "gcn":
            gnn = GCNConv
            gat_kwargs = dict(channels=channels, use_bias=add_bias)
        else:
            raise NotImplementedError(
                f"Gat type {gnn_type} not implemented, choose between gat, gatv2, gcn"
            )

        self.default_caching_device = _caching_device(self)
        self.gat_u_x = gnn(
            channels=self.channels,
            attn_heads=self.attn_heads,
            concat_heads=concat_heads,
            dropout_rate=0,
            activation="linear",
            return_attn_coef=return_attn_coef,
        )
        self.gat_r_x = gnn(
            channels=self.channels,
            attn_heads=self.attn_heads,
            concat_heads=concat_heads,
            dropout_rate=0,
            activation="linear",
            return_attn_coef=return_attn_coef,
        )
        self.gat_c_x = gnn(
            channels=self.channels,
            attn_heads=self.attn_heads,
            concat_heads=concat_heads,
            dropout_rate=0,
            activation="linear",
            return_attn_coef=return_attn_coef,
        )
        if self.h_gnn:
            self.gat_u_h = gnn(
                channels=self.channels,
                attn_heads=self.attn_heads,
                concat_heads=concat_heads,
                dropout_rate=0,
                activation="linear",
                return_attn_coef=return_attn_coef,
            )
            self.gat_r_h = gnn(
                channels=self.channels,
                attn_heads=self.attn_heads,
                concat_heads=concat_heads,
                dropout_rate=0,
                activation="linear",
                return_attn_coef=return_attn_coef,
            )
            self.gat_c_h = gnn(
                channels=self.channels,
                attn_heads=self.attn_heads,
                concat_heads=concat_heads,
                dropout_rate=0,
                activation="linear",
                return_attn_coef=return_attn_coef,
            )
        else:
            self.gat_u_h = l.Dense(self.channels, "linear")
            self.gat_r_h = l.Dense(self.channels, "linear")
            self.gat_c_h = l.Dense(self.channels, "linear")

    def build(self, input_shape):
        if self.add_bias:
            self.b_u = self.add_weight(
                shape=(self.channels,),
                initializer=init.Zeros,
                name="b_u",
                regularizer=self.regularizer,
                caching_device=self.default_caching_device,
            )
            self.b_r = self.add_weight(
                shape=(self.channels,),
                initializer=init.Zeros,
                name="b_r",
                regularizer=self.regularizer,
                caching_device=self.default_caching_device,
            )
            self.b_c = self.add_weight(
                shape=(self.channels,),
                initializer=init.zeros,
                name="b_c",
                regularizer=self.regularizer,
                caching_device=self.default_caching_device,
            )

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = tf.nest.flatten(inputs)
        h = states[0]
        if 0 < self.recurrent_dropout < 1:
            h_mask = self.get_recurrent_dropout_mask_for_cell(
                inputs=h, training=training, count=1
            )
            h = h * h_mask
        if 0 < self.dropout < 1:
            x_mask = self.get_dropout_mask_for_cell(
                inputs=x, training=training, count=1
            )
            x = x * x_mask
        xh = tf.concat([x, h], -1)
        if self.return_attn_coef:
            u_gat_x, u_attn_x = self.gat_u_x([x, a])
            r_gat_x, r_attn_x = self.gat_r_x([x, a])
            if self.h_gnn:
                u_gat_h, u_attn_h = self.gat_u_h([h, a])
                r_gat_h, r_attn_h = self.gat_r_h([h, a])
            else:
                u_gat_h = self.gat_u_h(h)
                r_gat_h = self.gat_r_h(h)
        else:
            u_gat_x = self.gat_u_x([x, a])
            r_gat_x = self.gat_r_x([x, a])
            if self.h_gnn:
                u_gat_h = self.gat_u_h([h, a])
                r_gat_h = self.gat_r_h([h, a])
            else:
                u_gat_h = self.gat_u_h(h)
                r_gat_h = self.gat_r_h(h)

        if self.add_bias:
            u = tf.nn.sigmoid(u_gat_x + u_gat_h + self.b_u)
            r = tf.nn.sigmoid(r_gat_x + r_gat_h + self.b_r)
        else:
            u = tf.nn.sigmoid(u_gat_x + u_gat_h)
            r = tf.nn.sigmoid(r_gat_x + r_gat_h)

        c_ = self.gat_c_x([x, a])
        if self.h_gnn:
            c_ = c_ + self.gat_c_h([r * h, a])
        else:
            c_ = c_ + self.gat_c_h(r * h)
        if self.add_bias:
            c_ = c_ + self.b_c

        c = tf.nn.tanh(c_)
        h_prime = u * h + (1 - u) * c

        if self.layer_norm:
            h_prime = self.ln(h_prime)
        if self.return_attn_coef:
            if self.h_gnn:
                return (h_prime, u_attn_x, r_attn_x, u_attn_h, r_attn_h), h_prime
            else:
                return (h_prime, u_attn_x, r_attn_x), h_prime
        else:
            return h_prime, h_prime

    def get_config(self):
        config = {
            "nodes": self.tot_nodes,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "attn_heads": self.attn_heads,
            "channels": self.channels,
            "concat_heads": self.concat_heads,
            "add_bias": self.add_bias,
            "activation": self.activation,
            "regularizer": self.regularizer,
            "return_attn_coef": self.return_attn_coef,
            "layer_norm": self.layer_norm,
            "initializer": (
                self.initializer
                if type(self.initializer) is str
                else tf.keras.utils.serialize_keras_object(self.initializer)
            ),
            "gnn_type": self.gnn_type,
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(
   "GNNRNN", "AttentionRNNCell"
)
class NestedGRUAttentionCell(DropoutRNNCellMixin, l.Layer):
    def __init__(
        self,
        nodes,
        dropout,
        recurrent_dropout,
        hidden_size_out,
        activation,
        regularizer=None,
        layer_norm=False,
        attn_heads=4,
        concat_heads=False,
        return_attn_coef=False,
        **kwargs,
    ):
        super(NestedGRUAttentionCell, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.hidden_size_out = hidden_size_out
        self.activation = activation
        self.recurrent_dropout = recurrent_dropout
        self.dropout = dropout
        self.regularizer = regularizer
        self.layer_norm = layer_norm
        self.return_attn_coef = return_attn_coef
        self.attn_heads = attn_heads
        self.output_size = tf.TensorShape((self.tot_nodes, self.hidden_size_out))
        self.output_size = tf.TensorShape((self.tot_nodes, self.hidden_size_out))
        if self.layer_norm:
            self.ln = l.LayerNormalization()
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

        self.gat_u = SelfAttention(
            channels=hidden_size_out,
            attn_heads=attn_heads,
            concat_heads=concat_heads,
            dropout_rate=0,
            activation=self.activation,
        )
        self.gat_r = SelfAttention(
            channels=hidden_size_out,
            attn_heads=attn_heads,
            concat_heads=concat_heads,
            dropout_rate=0,
            activation=self.activation,
        )
        self.gat_c = SelfAttention(
            channels=hidden_size_out,
            attn_heads=attn_heads,
            concat_heads=concat_heads,
            dropout_rate=0,
            activation=self.activation,
        )

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.b_u = self.add_weight(
            shape=(self.hidden_size_out,),
            initializer=init.Zeros,
            name="b_u",
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self.b_r = self.add_weight(
            shape=(self.hidden_size_out,),
            initializer=init.Zeros,
            name="b_r",
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self.b_c = self.add_weight(
            shape=(self.hidden_size_out,),
            initializer=init.zeros,
            name="b_c",
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = tf.nest.flatten(inputs)
        h = states[0]
        if 0 < self.recurrent_dropout < 1:
            h_mask = self.get_recurrent_dropout_mask_for_cell(
                inputs=h, training=training, count=1
            )
            h = h * h_mask
        if 0 < self.dropout < 1:
            x_mask = self.get_dropout_mask_for_cell(
                inputs=x, training=training, count=1
            )
            x = x * x_mask
        u_gat = self.gat_u(tf.concat([x, h], -1), a)
        r_gat = self.gat_r(tf.concat([x, h], -1), a)
        if self.add_bias:
            u_gat = self.b_u + u_gat
            r_gat = self.b_r + r_gat
        u = tf.nn.sigmoid(u_gat)
        r = tf.nn.sigmoid(r_gat)
        c_gat = self.gat_c(tf.concat([x, r * h], -1), a)
        if self.add_bias:
            c_gat = self.b_c + c_gat
        c = tf.nn.tanh(c_gat)
        h_prime = u * h + (1 - u) * c
        if self.layer_norm:
            h_prime = self.ln(h_prime)
        return h_prime, h_prime

    def get_config(self):
        config = {
            "nodes": self.tot_nodes,
            "hidden_size_out": self.hidden_size_out,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "regularizer": self.regularizer,
            "attn_heads": self.attn_heads,
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(NestedGRUAttentionCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchMultiBilinearDecoderDense(l.Layer):
    """
    inputs:
        - X of shape batch x N x d
    outputs: A of shape batch x N x N x R
    """

    def __init__(
        self,
        activation="relu",
        depth=2,
        regularizer="l2",
        diagonal=False,
        mask_diag=True,
        mask_diag_val=-1e8,
        **kwargs,
    ):
        super(BatchMultiBilinearDecoderDense, self).__init__(**kwargs)
        self.activation = activation
        self.regularizer = regularizer
        self.depth = depth
        self.mask_diag = mask_diag
        self.mask_diag_val = mask_diag_val
        self.diagonal = diagonal

    def build(self, input_shape):
        x = input_shape
        if self.diagonal:
            self.R = self.add_weight(
                shape=(
                    self.depth,
                    x[-1],
                ),
                initializer="glorot_normal",
                regularizer=self.regularizer,
                name="bilinear_matrix",
            )
            self.R = tf.linalg.diag(self.R)
            self.R = tf.transpose(self.R, perm=(1, 2, 0))
        else:
            self.R = self.add_weight(
                shape=(x[-1], x[-1], self.depth),
                initializer="glorot_normal",
                regularizer=self.regularizer,
                name="bilinear_matrix",
            )
        self.diag = tf.constant(tf.linalg.diag([tf.ones(x[-2])]) * self.mask_diag_val)[
            ..., None
        ]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        A = tf.einsum("...Bd,dor,...Po->...BPr", x, self.R, x)
        if self.mask_diag:
            return A + self.diag
        A = act.get(self.activation)(A)
        return A


class BilinearDecoderSparse(l.Layer):
    def __init__(self, activation="relu", diagonal=False, **kwargs):
        super(BilinearDecoderSparse, self).__init__(**kwargs)
        self.initializer = init.GlorotNormal()
        self.diagonal = diagonal
        self.activation = activation

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        if self.diagonal:
            self.R_kernel = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1]))
            )
            self.R_kernel = tf.linalg.diag(self.R_kernel)
        else:
            self.R_kernel = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1], X_shape[-1]))
            )

    def call(self, inputs, **kwargs):
        X, A = inputs
        i, j = A.indices[:, 0], A.indices[:, 1]
        e1 = tf.gather(X, i)
        e2 = tf.gather(X, j)
        left = tf.einsum("ij,jk->ik", e1, self.R_kernel)
        right = tf.einsum("ij,ij->i", left, e2)
        if self.activation:
            A_pred = act.get(self.activation)(right)
        A_pred = tf.sparse.SparseTensor(A.indices, A_pred, A.shape)
        return X, A_pred


class SelfAttention(l.Layer):
    def __init__(
        self,
        channels=10,
        attn_heads=5,
        dropout_rate=0.5,
        activation="relu",
        concat_heads=False,
        return_attn_coef=False,
        renormalize=False,
        initializer="glorot_normal",
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.concat_heads = concat_heads
        self.return_attn_coef = return_attn_coef
        self.renormalize = renormalize
        self.initializer = init.get(initializer)

    def build(self, input_shape):
        """
        Inputs: X, A
            - X: shape(NxTxd)
            #- A: shape(TxNxN)
            - A: shape(NxTxT)
        """
        x, a = input_shape
        self.q_w = self.add_weight(
            name="query",
            shape=(self.attn_heads, x[-1], self.channels),
            initializer=self.initializer,
        )
        self.k_w = self.add_weight(
            name="key",
            shape=(self.attn_heads, x[-1], self.channels),
            initializer=self.initializer,
        )
        self.v_w = self.add_weight(
            name="value",
            shape=(self.attn_heads, x[-1], self.channels),
            initializer=self.initializer,
        )
        if self.dropout_rate:
            self.drop = l.Dropout(self.dropout_rate)

    def call(self, inputs, mask, *args, **kwargs):
        """
        query=key=value:
            - n: nodes V batch size
            - t: time dim if time series or number of nodes if n=batch size
            - d: input embedding dimension
            - o: output embedding dimension
            - h: number of heads
        x=input embedding of shape NxTxd
        a=input adjacency matrix of shape NxTxT
            -
        """
        x = inputs
        a = mask
        query = tf.einsum("ntd,hdo->ntho", x, self.q_w)
        key = tf.einsum("ntd,hdo->ntho", x, self.k_w)
        value = tf.einsum("ntd,hdo->ntho", x, self.v_w)
        qk = tf.einsum("ntho,nzho->nhtz", query, key)  # NxHxTxT
        qk /= tf.sqrt(tf.cast(self.channels, tf.float32))
        if mask is not None:
            qk += tf.transpose(
                [tf.where(a == 0.0, -1e10, 0.0)] * self.attn_heads, perm=(1, 0, 2, 3)
            )  # NxHxTxT
        soft_qk = tf.nn.softmax(qk, axis=-1)
        if self.dropout_rate:
            soft_qk = self.drop(soft_qk)
            if self.renormalize:
                soft_qk = tf.nn.softmax(soft_qk, axis=-1)
        x_prime = tf.einsum("nhtz,nzho->nhto", soft_qk, value)
        if self.concat_heads:
            x_prime = tf.transpose(x_prime, (0, 2, 1, 3))  # NxTxHxO
            x_prime = tf.reshape(x_prime, (*tf.shape(x_prime)[:-2], -1))  # NxTxHO
        else:
            x_prime = tf.reduce_mean(x_prime, axis=1)
            x_prime = tf.squeeze(x_prime)  # NxTxO
        if self.return_attn_coef:
            return x_prime, soft_qk
        return x_prime


@tf.keras.utils.register_keras_serializable(package="GNN", name="GATv2Layer")
class GATv2Layer(l.Layer):
    def __init__(
        self,
        attn_heads,
        channels,
        concat_heads=False,
        add_bias=True,
        activation="relu",
        dropout_rate=0,
        residual=False,
        initializer=init.GlorotNormal(seed=0),
        regularizer=None,
        return_attn_coef=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = attn_heads
        self.channels = channels
        self.concatenate_output = concat_heads
        self.add_bias = add_bias
        self.activation = act.get(activation)
        self.residual = residual
        self.initializer = init.get(initializer)
        self.regularizer = regu.get(regularizer)
        self.return_attention = return_attn_coef
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout_attn = l.Dropout(dropout_rate)
            self.dropout_feat = l.Dropout(dropout_rate)

    def build(self, input_shape):
        caching_device = _caching_device(self)
        x, a = input_shape
        self.w_shape = (self.heads, self.channels, self.channels)
        self.attn_shape = (self.heads, self.channels)
        self.W_self = self.add_weight(
            name=f"kern_features_self",
            shape=self.w_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            caching_device=caching_device,
        )
        self.W_ngb = self.add_weight(
            name=f"kern_features_ngb",
            shape=self.w_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            caching_device=caching_device,
        )
        self.attn = self.add_weight(
            name=f"kern_attention",
            shape=self.attn_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            caching_device=caching_device,
        )
        self.kernel = self.add_weight(
            name="kernel_feature",
            shape=(x[-1], self.channels),
            caching_device=caching_device,
        )
        if self.add_bias:
            self.bias = self.add_weight(
                name="kern_bias",
                shape=(1, *self.attn_shape),
                caching_device=caching_device,
                initializer=init.Zeros(),
            )
            self.bias0 = self.add_weight(
                name="kern_bias_i",
                shape=(self.channels,),
                caching_device=caching_device,
                initializer=init.Zeros(),
            )

    def call(self, inputs, training=None, mask=None):
        """
        When the adjacency matrix is a Sparse Tensor batch size is not supported
        :param inputs: Tuple of features NxF and Sparse Adjacency Matrix NxN, in dense mode a tuple of BxNxF and Dense
        Adjacency Matrix BxNxN
        :param training: Whether in training mode
        :param mask: Not Used
        :return: Updated Features of shape Nx(HF) or NF
        """
        x, a = inputs
        assert a.shape[-1] == a.shape[-2]
        x = x @ self.kernel
        if self.add_bias:
            x = x + self.bias0
        x = act.get(self.activation)(x)
        if self.dropout > 0:
            x = self.dropout_feat(x)

        if isinstance(a, tf.sparse.SparseTensor):
            tf.assert_rank(a, 2)
            N = tf.shape(x, out_type=a.indices.dtype)[-2]
            i, j = a.indices[:, 0], a.indices[:, 1]
            x_i_prime = tf.einsum("NF,HFO->NHO", x, self.W_self)
            x_i = tf.gather(x_i_prime, i, axis=0)
            x_j_prime = tf.einsum("NF,HFO->NHO", x, self.W_ngb)
            x_j = tf.gather(x_j_prime, j, axis=0)
            x_ij_prime = x_i + x_j  # EHO
            if self.add_bias:
                x_ij_prime = x_ij_prime + self.bias
            x_ij_prime = self.activation(x_ij_prime)
            a_ij = tf.einsum("EHO,HO->EH", x_ij_prime, self.attn)
            a_soft_ij = unsorted_segment_softmax(a_ij, j, N)
            if 0 < self.dropout < 1:
                a_soft_ij = self.dropout_attn(a_soft_ij)
            out = a_soft_ij[..., None] * x_i[:, None]  # EH
            out = tf.math.unsorted_segment_sum(out, j, N)  # NHF
            if self.concatenate_output:
                out = tf.reshape(out, (-1, self.attn_shape[0] * self.attn_shape[1]))
            else:
                out = tf.math.reduce_mean(out, -2)
            if self.return_attention:
                return out, a_soft_ij
            else:
                return out
        else:
            x_i = tf.einsum("...NF,HFO->...HON", x, self.W_self)
            x_j = tf.einsum("...NF,HFO->...HON", x, self.W_ngb)
            x_ij = x_i[..., None, :] + x_j[..., None]  # BHONN
            if self.add_bias:
                x_ij = x_ij + self.bias[:, :, :, None, None]
            x_ij_activated = self.activation(x_ij)
            e_ij = tf.einsum("...HONK,HO->...HNK", x_ij_activated, self.attn)
            a_mask = tf.where(a == 0, -10e9, 0.0)
            a_mask = tf.repeat(a_mask[:, None, ...], self.heads, 1)  # BHNN
            a_soft_ij = tf.nn.softmax(a_mask + e_ij)
            if 0 < self.dropout < 1:
                a_soft_ij = self.dropout_attn(a_soft_ij)
            x_prime = tf.einsum("...HNK,...NF->...KHF", a_soft_ij, x)
            if self.concatenate_output:
                out = tf.reshape(
                    x_prime, (*x.shape[:2], self.heads * x.shape[-1])
                )  # BxNx(FH)
            else:
                out = tf.reduce_mean(x_prime, 2)  # BxNxF (reduce over heads)
            if self.return_attention:
                return out, a_soft_ij
            else:
                return out

    def get_config(self):
        config = {
            "attn_heads": self.heads,
            "channels": self.channels,
            "concat_heads": self.concatenate_output,
            "add_bias": self.add_bias,
            "activation": act.serialize(self.activation),
            "dropout_rate": self.dropout,
            "residual": self.residual,
            "initializer": init.serialize(self.initializer),
            "regularizer": regu.serialize(self.regularizer),
            "return_attn_coef": self.return_attention,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ScaleLogProb(tfd.Distribution):
    def __init__(
        self,
        distribution: tfd.Distribution,
        weight,
        validate_args=False,
        allow_nan_stats=True,
        name=None,
    ):
        parameters = dict(locals())
        with tf.name_scope(name or distribution.name) as name:
            self._distribution = distribution
            self._weight = tensor_util.convert_nonref_to_tensor(
                weight, dtype_hint=tf.float32
            )
            super().__init__(
                dtype=distribution.dtype,
                reparameterization_type=distribution.reparameterization_type,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name,
            )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            distribution=parameter_properties.BatchedComponentProperties(),
            weight=parameter_properties.ParameterProperties(
                shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED
            ),
        )

    def _event_shape(self):
        return self.distribution.event_shape

    def _event_shape_tensor(self):
        return self.distribution.event_shape_tensor()

    def _log_prob(self, x):
        return self.weight * self.distribution.log_prob(x)
