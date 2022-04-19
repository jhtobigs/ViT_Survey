import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, LayerNormalization, Flatten
from model.vit import MLPBlock
from transformer.transformer import MultiHeadAttention

"""
To Do 
1. Norm같은 디테일 논문과 같이 구현
2. Docstring 추가
"""


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


class WindowAttention(Layer):
    def __init__(self, shift_size, input_resolution, dims=96, window_size=7):
        super().__init__()
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.dims = dims
        self.num_heads = dims // 32
        self.bias_attention = BiasMultiHeadAttention(self.num_heads, dims, window_size)
        if shift_size > 0:
            self.attn_mask = self._make_swin_mask()
        else:
            self.attn_mask = None

    def _make_swin_mask(self):
        img_mask = np.zeros((1, self.input_resolution, self.input_resolution, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask)

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = tf.reshape(mask_windows, (-1, self.window_size * self.window_size))

        attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
        attn_mask = tf.where(attn_mask == 0, x=0.0, y=-1e3)
        return attn_mask[:, tf.newaxis]

    def call(self, x):
        B, H, W, C = tf.shape(x)
        if self.shift_size > 0:
            x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=[1, 2])
        x = window_partition(x, self.window_size)
        x = tf.reshape(x, (-1, self.window_size * self.window_size, C))
        x = self.bias_attention(x, x, x, mask=self.attn_mask)
        x = tf.reshape(x, (-1, self.window_size, self.window_size, C))
        x = window_reverse(x, self.window_size, self.input_resolution, self.input_resolution)
        if self.shift_size > 0:
            x = tf.roll(x, shift=(self.shift_size, self.shift_size), axis=[1, 2])
        return x


class SwinTransformerBlock(Layer):
    def __init__(self, input_resolution, dims=96, window_size=7):
        super().__init__()
        self.input_resolution = input_resolution
        self.ln1 = LayerNormalization(epsilon=1e-5)
        self.window_attention = WindowAttention(0, input_resolution, dims, window_size)
        self.ln2 = LayerNormalization(epsilon=1e-5)
        self.mlp1 = MLPBlock(ffn_dims=dims * 4, dims=dims)

        self.ln3 = LayerNormalization(epsilon=1e-5)
        self.swin_attention = WindowAttention(window_size // 2, input_resolution, dims, window_size)
        self.ln4 = LayerNormalization(epsilon=1e-5)
        self.mlp2 = MLPBlock(ffn_dims=dims * 4, dims=dims)

    def call(self, x):
        y = self.ln1(x)
        y = self.window_attention(y)
        x = x + y

        y = self.ln2(x)
        y = self.mlp1(x)
        x = x + y

        y = self.ln3(x)
        y = self.swin_attention(y)
        x = x + y

        y = self.ln4(x)
        y = self.mlp2(x)
        x = x + y
        return x


class PatchMerging(Layer):
    def __init__(self, dims):
        super().__init__()
        self.ln = LayerNormalization(epsilon=1e-6)
        self.merge_dense = Dense(units=dims, use_bias=False)

    def call(self, x):
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.ln(x)
        x = self.merge_dense(x)
        return x


class SwinTransformer(Model):
    def __init__(self, image_size, C, num_blocks: list, window_size=7):
        super().__init__()
        patch_size = [4, 8, 16, 32]
        dims = [C, 2 * C, 4 * C, 8 * C]
        input_resolution = [image_size // patch for patch in patch_size]
        self.patch_encoder = PatchEncoder(image_size, dims[0], patch_size[0])
        for i, num_iter in enumerate(num_blocks):
            if i == 0:
                self.blocks = [SwinTransformerBlock(input_resolution[0], dims[0], window_size)]
                continue
            self.blocks.append(PatchMerging(dims[i]))
            for _ in range(num_iter // 2):
                self.blocks.append(SwinTransformerBlock(input_resolution[i], dims[i], window_size))

    def call(self, x):
        x = self.patch_encoder(x)
        for block in self.blocks:
            x = block(x)
        return x
