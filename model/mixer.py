import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from model.vit import PatchPartition

# TODO Add Stochastic Depth & Docstring
class PatchEncoder(Layer):
    def __init__(self, patch_size: int, dims: int):
        super().__init__()
        self.patch_split = PatchPartition(patch_size)
        self.patch_projection = Dense(dims)

    def call(self, x):
        x = self.patch_split(x)
        x = self.patch_projection(x)
        return x


class MLPBlock(Layer):
    def __init__(self, ffn_dims=2048, dims=512, dropout_rate=0.1, activation="gelu", **kwargs):
        super().__init__()
        self.fc1 = Dense(units=ffn_dims, activation=activation)
        self.dropout1 = Dropout(rate=dropout_rate)
        self.fc2 = Dense(units=dims)
        self.dropout2 = Dropout(rate=dropout_rate)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class MixerLayer(Layer):
    def __init__(self, dims: int, num_patches: int, dims_s: int, dims_c: int, dropout_rate=0.0):
        super().__init__()
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.mlp_patch = MLPBlock(ffn_dims=dims_s, dims=num_patches, dropout_rate=dropout_rate)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.mlp_channel = MLPBlock(ffn_dims=dims_c, dims=dims, dropout_rate=dropout_rate)

    def call(self, x):
        y = self.ln1(x)
        y = tf.transpose(y, (0, 2, 1))
        y = self.mlp_patch(y)
        y = tf.transpose(y, (0, 2, 1))
        x = x + y
        y = self.ln2(x)
        y = self.mlp_channel(y)
        x = x + y
        return x


class PredictionHead(Layer):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.pre_ln = LayerNormalization(epsilon=1e-6)
        self.gap = GlobalAveragePooling1D()
        self.head = Dense(num_classes, name="predict_head", activation="softmax")

    def call(self, x):
        x = self.pre_ln(x)
        x = self.gap(x)
        x = self.head(x)
        return x


class MLPMixer(Model):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dims: int,
        dims_s: int,
        dims_c: int,
        num_layer: int,
        dropout_rate=0.0,
        include_top=False,
        num_classes=1000,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size should be separated by patch_size."
        num_patches = (image_size**2) // (patch_size**2)
        self.include_top = include_top
        self.patch_encoder = PatchEncoder(patch_size, dims)
        self.mixer_layer = [
            MixerLayer(dims, num_patches, dims_s, dims_c, dropout_rate) for i in range(num_layer)
        ]
        if include_top:
            self.prediction_head = PredictionHead(num_classes)

    def call(self, x):
        x = self.patch_encoder(x)
        for mixer in self.mixer_layer:
            x = mixer(x)
        if self.include_top:
            x = self.prediction_head(x)
        return x
