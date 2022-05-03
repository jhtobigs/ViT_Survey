from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization


class PatchEncoder(Layer):
    def __init__(self, image_size, dims, patch_size=7):
        super().__init__()
        assert image_size % patch_size == 0, "image_size should be separated by patch_size."
        patch_tuple = (patch_size, patch_size)
        self.image_size = image_size
        self.embedding = Conv2D(filters=dims, kernel_size=patch_tuple, strides=patch_tuple, activation="gelu")
        self.bn = BatchNormalization()

    def call(self, x):
        x = self.embedding(x)
        x = self.bn(x)
        return x
