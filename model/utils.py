import math
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


def position_embedding_sine(
    attention_mask, num_pos_features=256, temperature=10000.0, normalize=True, scale=2 * math.pi
):
    """Sine-based positional embeddings for 2D images.
    Args:
      attention_mask: a `bool` Tensor specifying the size of the input image to
        the Transformer and which elements are padded, of size [batch_size,
        height, width]
      num_pos_features: a `int` specifying the number of positional features,
        should be equal to the hidden size of the Transformer network
      temperature: a `float` specifying the temperature of the positional
        embedding. Any type that is converted to a `float` can also be accepted.
      normalize: a `bool` determining whether the positional embeddings should be
        normalized between [0, scale] before application of the sine and cos
        functions.
      scale: a `float` if normalize is True specifying the scale embeddings before
        application of the embedding function.
    Returns:
      embeddings: a `float` tensor of the same shape as input_tensor specifying
        the positional embeddings based on sine features.
    """
    if num_pos_features % 2 != 0:
        raise ValueError(
            "Number of embedding features (num_pos_features) must be even when "
            "column and row embeddings are concatenated."
        )
    num_pos_features = num_pos_features // 2

    # Produce row and column embeddings based on total size of the image
    # <tf.float>[batch_size, height, width]
    attention_mask = tf.cast(attention_mask, tf.float32)
    row_embedding = tf.cumsum(attention_mask, 1)
    col_embedding = tf.cumsum(attention_mask, 2)

    if normalize:
        eps = 1e-6
        row_embedding = row_embedding / (row_embedding[:, -1:, :] + eps) * scale
        col_embedding = col_embedding / (col_embedding[:, :, -1:] + eps) * scale

    dim_t = tf.range(num_pos_features, dtype=row_embedding.dtype)
    dim_t = tf.pow(temperature, 2 * (dim_t // 2) / num_pos_features)

    # Creates positional embeddings for each row and column position
    # <tf.float>[batch_size, height, width, num_pos_features]
    pos_row = tf.expand_dims(row_embedding, -1) / dim_t
    pos_col = tf.expand_dims(col_embedding, -1) / dim_t
    pos_row = tf.stack([tf.sin(pos_row[:, :, :, 0::2]), tf.cos(pos_row[:, :, :, 1::2])], axis=4)
    pos_col = tf.stack([tf.sin(pos_col[:, :, :, 0::2]), tf.cos(pos_col[:, :, :, 1::2])], axis=4)

    # final_shape = pos_row.shape.as_list()[:3] + [-1]
    final_shape = list(tf.shape(pos_row)[:3]) + [-1]
    pos_row = tf.reshape(pos_row, final_shape)
    pos_col = tf.reshape(pos_col, final_shape)
    output = tf.concat([pos_row, pos_col], -1)

    embeddings = tf.cast(output, tf.float32)
    return embeddings


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


def load_backbone(backbone_name) -> Model:
    if backbone_name == "resnet50":
        model = tf.keras.applications.ResNet50(include_top=False)
    return model
