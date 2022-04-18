import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, LayerNormalization
from transformer.transformer import MultiHeadAttention


class PatchPartion(Layer):
    """Splits Image into Patches

    Attributes :
        patch_size : he is size of square patches (patch_size, patch_size)
    """

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def split_patches(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        B, _, _, D = tf.shape(patches)
        patches = tf.reshape(patches, [B, -1, D])
        return patches

    def call(self, images):
        """Splits Images into Patches

        Args :
            images : A 4-D Tensor with shape [batch, in_rows, in_cols, depth].

        Returns
            A 3-D Tensor with shape ([batch, num_patches, dimension])
        """
        patches = self.split_patches(images=images)
        return patches


# ToDo Docstrings
class PatchEmbedding(Layer):
    def __init__(self, dims: int):
        super().__init__()
        self.linear_projection = Dense(units=dims)

    def call(self, patches):
        patches = self.linear_projection(patches)
        return patches


# ToDo Docstrings
class PositionalEmbedding(Layer):
    """Learnable Positional Embedding for tokens"""

    def __init__(self, num_patches: int, dims: int):
        super().__init__()
        self.num_patches = num_patches
        self.embedding = Embedding(input_dim=num_patches + 2, output_dim=dims)

    def call(self, patches):
        B, N, D = tf.shape(patches)
        position = tf.range(0, N)
        pos_embedding = self.embedding(position)[tf.newaxis]
        pos_embedding = tf.repeat(pos_embedding, B, axis=0)
        return pos_embedding


# ToDo Docstrings
class PatchEncoder(Layer):
    def __init__(self, patch_size: int, num_patches: int, dims: int):
        super().__init__()
        self.patch_split = PatchPartion(patch_size)
        self.patch_embedding = PatchEmbedding(dims)
        self.position_embedding = PositionalEmbedding(num_patches, dims)

    def _append_class_token(self, patches):
        class_token = tf.zeros_like(patches[:, :1, :])
        patches = tf.concat([class_token, patches], axis=1)
        return patches

    def call(self, x):
        x = self.patch_split(x)
        x = self.patch_embedding(x)
        x = self._append_class_token(x)
        x += self.position_embedding(x)
        return x


class MLPBlock(Layer):
    def __init__(self, ffn_dims=2048, dims=512, dropout_rate=0.1, activation="gelu"):
        super().__init__()
        self.fc1 = Dense(units=ffn_dims, activation=activation)
        self.dropout1 = Dropout(rate=dropout_rate)
        self.fc2 = Dense(units=dims, activation=activation)
        self.dropout2 = Dropout(rate=dropout_rate)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
