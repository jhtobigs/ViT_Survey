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
