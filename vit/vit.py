import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding


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
