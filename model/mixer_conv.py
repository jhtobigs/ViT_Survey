import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    BatchNormalization,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Dense,
)


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


class ConvMixerLayer(Layer):
    def __init__(self, dims, kernel_size):
        super().__init__()
        self.depthwise = DepthwiseConv2D(
            kernel_size=kernel_size, strides=1, padding="same", activation="gelu"
        )
        self.bn1 = BatchNormalization()
        self.pointwise = Conv2D(filters=dims, kernel_size=1, activation="gelu")
        self.bn2 = BatchNormalization()

    def call(self, x):
        y = self.depthwise(x)
        y = self.bn1(y)
        x = x + y
        x = self.pointwise(x)
        x = self.bn2(x)
        return x


class PredictionHead(Layer):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.gap = GlobalAveragePooling2D()
        self.head = Dense(num_classes, name="predict_head", activation="softmax")

    def call(self, x):
        x = self.gap(x)
        x = self.head(x)
        return x


class ConvMixer(Model):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        kernel_size: int,
        dims: int,
        num_layer: int,
        include_top=False,
        num_classes=1000,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size should be separated by patch_size."
        self.num_patches = (image_size**2) // (patch_size**2)
        self.include_top = include_top
        self.patch_encoder = PatchEncoder(patch_size, dims)
        self.mixer_layer = [ConvMixerLayer(dims, kernel_size) for _ in range(num_layer)]
        if include_top:
            self.prediction_head = PredictionHead(num_classes)

    def call(self, x):
        x = self.patch_encoder(x)
        for mixer in self.mixer_layer:
            x = mixer(x)
        if self.include_top:
            x = self.prediction_head(x)
        return x
