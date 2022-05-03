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
