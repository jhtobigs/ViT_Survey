import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


def get_radians(len_position: int, dims: int):
    pos = np.arange(len_position)[:, np.newaxis]
    i = np.arange(dims)[np.newaxis, :]
    exponent = (2 * (i // 2)) / dims
    radians = pos / np.power(10000.0, exponent)
    return radians


def make_sine_pos_encoding(len_position: int, dims: int) -> np.array:
    radians = get_radians(len_position, dims)
    radians[:, 0::2] = np.sin(radians[:, 0::2])
    radians[:, 1::2] = np.cos(radians[:, 1::2])
    radians = radians.astype("float32")
    return radians


def create_padding_mask(inputs):
    """
    Args :
        inputs : Input Sequence (B,N)
    Returns:
        tf.float32 : Padding mask for Multi-Head-Attention (B,1,1,N)
    """
    padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]  # B, 1, 1, N


def create_look_ahead_mask(seq_len: int):
    """
    Args :
        seq_len : Sequence Length
    Returns :
        tf.float32 : Look-Ahead Mask (N,N)
    """
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask


def create_padding_lookahed_mask(inputs):
    padding_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(tf.shape(inputs)[1])
    mask = tf.maximum(look_ahead_mask, padding_mask)
    return mask


def load_backbone(self, backbone_name) -> Model:
    return None
