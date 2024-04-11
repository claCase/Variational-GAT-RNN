from typing import Type
import tensorflow as tf
from .utils import normalize_adj, node_degree, positive_variance
from tensorflow_probability.python.distributions import LogNormal, HalfNormal

klo = tf.keras.losses


def custom_mse(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))


def min_cut(A, S, normalize=True):
    """
    arxiv.org/abs/1907.00481
    S: ...NxC
    A: ...NxN
    """
    if normalize:
        A = normalize_adj(A, symmetric=True)
    D = node_degree(A, symmetric=True)
    num = tf.linalg.trace(tf.einsum("...ji,...jk,...kl->...il", S, A, S))
    den = tf.linalg.trace(tf.einsum("...ji,...jk,...kl->...il", S, D, S))
    mc = -num / den  # mincut loss

    k = tf.cast(S.shape[-1], S.dtype)
    I = tf.eye(k, dtype=S.dtype) / tf.sqrt(k)
    innerS = tf.einsum("...ji,...jk->...ik", S, S)
    normS = tf.linalg.norm(innerS, axis=(-1, -2))
    ort = tf.linalg.norm(
        innerS / normS[..., None, None] - I, axis=(-1, -2)
    )  # orthogonality constraint

    return mc + ort


def zero_inflated_nlikelihood(labels: Type[tf.Tensor],
                             logits: Type[tf.Tensor],
                             mean_axis=(-2, -1),
                             pos_weight=1.,
                             smooth=0.0,
                             distribution="lognormal") -> Type[tf.Tensor]:
    """
    Computes the zero inflated lognormal loss.
    Arguments:
    :param labels: True targets, tensor of shape [batch_size, 1].
    :param logits: Logits of output layer, tensor of shape [batch_size, 3] -> prob, mean, var
    :param mean_axis: Axis of indipendent joint distribution
    :param pos_weight: scalar indicating positive class re-weighting
    :param smooth: label smoothing scalar parameter
    :param distribution: Either Lognormal or Halfnormal
    """
    if distribution == "lognormal":
        assert tf.debugging.assert_equal(tf.shape(logits)[-1], 3)
    elif distribution == "halfnormal":
        assert tf.debugging.assert_equal(tf.shape(logits)[-1], 2)
    else:
        raise NotImplementedError(f"Distribution {distribution} is not supported, choose from lognormal or halfnormal")
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    positive = tf.cast(labels > 0, tf.float32)
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    logits.shape.assert_is_compatible_with(
        tf.TensorShape(labels.shape[:-1].as_list() + [3]))

    positive_logits = logits[..., :1]
    positive_logits = tf.nn.sigmoid(positive_logits)
    classification_loss = tf.keras.losses.binary_crossentropy(
        y_true=positive, y_pred=positive_logits, from_logits=False, axis=-1, label_smoothing=smooth)
    positive0 = tf.squeeze(positive, -1)
    classification_loss = classification_loss * (1 - positive0) + positive0 * classification_loss * pos_weight
    classification_loss = tf.reduce_mean(classification_loss, axis=mean_axis)
    loc = logits[..., 1:2]
    # loc = tf.math.maximum(tf.nn.relu(loc), tf.math.sqrt(K.epsilon()))
    safe_labels = positive * labels + (
            1 - positive) * tf.ones_like(labels)
    if distribution == "lognormal":
        scale = positive_variance(logits[..., 2:])
        regression_loss = -tf.reduce_mean(
            tf.squeeze(positive * LogNormal(loc=loc, scale=scale).log_prob(safe_labels), -1),
            axis=mean_axis)
    else:
        regression_loss = -tf.reduce_mean(tf.squeeze(
            positive * HalfNormal(scale=loc).log_prob(safe_labels), -1),
            axis=mean_axis)
    return 0.5 * classification_loss + 0.5 * regression_loss
