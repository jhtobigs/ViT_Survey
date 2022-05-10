from turtle import pos
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Embedding, Conv2D
from model.utils.utils import load_backbone, make_sine_pos_encoding, position_embedding_sine
from model.transformer import EncoderBlock, DecoderBlock


class Backbone(Model):
    def __init__(self, backbone_name: str, frozen_layer_type: Layer, dims, dilation=False):
        super().__init__()
        self.backbone = load_backbone(backbone_name)
        self.dilation = dilation
        if frozen_layer_type is not None:
            self._freeze_layer(frozen_layer_type)
        self.projection = Conv2D(filters=dims, kernel_size=1)

    def _freeze_layer(self, frozen_layer_type: Layer):
        for layer in self.backbone.layers:
            if isinstance(layer, frozen_layer_type):
                layer.trainable = False
        print(f"Freeze {frozen_layer_type} in Backbone")

    def _extend_resolution(self):
        pass

    def call(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x


class PredictFFN(Layer):
    def __init__(self, ffn_dims=2048, dims=512, activation="relu", last_activation=None):
        super().__init__()
        self.fc1 = Dense(units=ffn_dims, activation=activation)
        self.fc2 = Dense(units=ffn_dims, activation=activation)
        self.fc3 = Dense(units=dims, activation=last_activation)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DETR(Model):
    def __init__(
        self,
        backbone_name="resnet50",
        frozen_layer_type=BatchNormalization,
        dims=256,
        ffn_dims=2048,
        num_heads=8,
        num_class=1000,
        num_query=100,
        num_enc_block=6,
        num_dec_block=6,
        activation="relu",
        dropout_rate=0.1,
    ):
        super().__init__()
        self.dims = dims
        self.backbone = Backbone(backbone_name, frozen_layer_type, dims)
        self.object_query = self.add_weight(
            shape=[num_query, dims],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            dtype=tf.float32,
        )
        self.encoder = Sequential(
            [EncoderBlock(ffn_dims, num_heads, dims, activation, dropout_rate) for i in range(num_enc_block)]
        )
        self.decoder = [
            DecoderBlock(ffn_dims, num_heads, dims, activation, dropout_rate) for i in range(num_dec_block)
        ]
        self.pred_class = PredictFFN(ffn_dims, num_class + 1, last_activation="softmax")
        self.pred_bbox = PredictFFN(ffn_dims, 4, last_activation="sigmoid")

    def call(self, x):
        mask = tf.expand_dims(tf.cast(tf.not_equal(tf.reduce_sum(x, axis=-1), 0), x.dtype), axis=-1)
        x = self.backbone(x)
        B, N, H, D = x.shape
        mask = tf.image.resize(mask, x.shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        pos_embed = position_embedding_sine(mask[:, :, :, 0], num_pos_features=self.dims)
        pos_embed = tf.reshape(pos_embed, x.shape)
        x += pos_embed
        x = tf.reshape(x, (B, -1, D))
        enc_output = self.encoder(x)
        d_output = tf.repeat(self.object_query[tf.newaxis], B, 0)
        for layer in self.decoder:
            d_output = layer(d_output, enc_output, look_ahead_mask=None, mask=None)
        pred_class = self.pred_class(x)
        pred_bbox = self.pred_bbox(x)
        return pred_class, pred_bbox
