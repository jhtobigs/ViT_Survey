import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, LayerNormalization, Flatten
from transformer.transformer import MultiHeadAttention


def window_partition(x, window_size=7):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, C))

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (B, H, W, -1))
    return x


class PatchEncoder(Layer):
    def __init__(self, image_size, dims, patch_size=4):
        super().__init__()
        assert image_size % patch_size == 0, "image_size should be separated by patch_size."
        patch_tuple = (patch_size, patch_size)
        self.image_size = image_size
        self.partition_embedding = Conv2D(filters=dims, kernel_size=patch_tuple, strides=patch_tuple)

    def call(self, x):
        x = self.partition_embedding(x)
        return x


class BiasMultiHeadAttention(MultiHeadAttention):
    def __init__(self, num_heads, dims, window_size=7):
        super().__init__(num_heads=num_heads, dims=dims)
        self.window_size = window_size
        self.bias = self._make_bias_table()

    def _make_bias_table(self):
        initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)
        table_shape = ((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads)
        self.table = tf.Variable(initial_value=initializer(shape=table_shape))

        coords_h = tf.range(self.window_size)
        coords_w = tf.range(self.window_size)
        coord = tf.stack(tf.meshgrid(coords_w, coords_h)[::-1])  # 2, Wh, Ww
        coords_flatten = Flatten()(coord)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = tf.transpose(relative_coords, perm=(1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        h, w = tf.split(relative_coords, 2, axis=-1)
        h += self.window_size - 1
        h *= 2 * self.window_size - 1
        w += self.window_size - 1
        self.position_index = tf.reshape(tf.reduce_sum(tf.concat([h, w], axis=-1), axis=-1), -1)
        bias_shape = (self.window_size**2, self.window_size**2, -1)
        bias = tf.gather(params=self.table, indices=self.position_index)
        bias = tf.reshape(tensor=bias, shape=bias_shape)
        bias = tf.transpose(bias, perm=(2, 0, 1))[tf.newaxis]
        return bias

    def scale_dot_product(self, q, k, v, mask=None):
        x = tf.matmul(q, k, transpose_b=True)
        x /= tf.math.sqrt(self.head_dims)
        x += self.bias
        if mask is not None:
            B = tf.shape(x)[0]
            W = tf.shape(mask)[0]
            mask = tf.repeat(mask, B // W, axis=0)
            x += mask
        x = tf.nn.softmax(x, axis=-1)
        x = tf.matmul(x, v)
        return x
