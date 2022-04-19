import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, LayerNormalization, Flatten


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
