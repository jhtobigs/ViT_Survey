from requests import patch
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, Embedding, LayerNormalization
from model.transformer import EncoderBlock
from model.vit import PositionalEmbedding
from model.utils.mask_generator import MaskingGenerator


class BEiTPatchEncoder(Layer):
    def __init__(self, patch_size: int, num_patches: int, dims: int):
        super().__init__()
        self.dims = dims
        self.cls_token = tf.Variable(tf.random.truncated_normal(shape=(1, 1, dims)), trainable=True)
        self.mask_token = tf.Variable(tf.random.truncated_normal(shape=(1, 1, dims)), trainable=True)
        self.patch_masking = MaskingGenerator(patch_size, num_masking_patches=int(num_patches * 0.4))
        self.patch_embedding = Conv2D(filters=dims, kernel_size=patch_size, strides=patch_size)
        self.position_embedding = PositionalEmbedding(num_patches, dims)

    def _blockwise_masking(self, patches, batch_size: int):
        mask_patch = self.patch_masking(batch_size)[..., tf.newaxis]
        patches = patches * (1 - mask_patch) + (self.mask_token * mask_patch)
        return patches

    def _process_token(self, patches):
        B = tf.shape(patches)[0]
        patches = self._blockwise_masking(patches, B)
        patches = tf.reshape(patches(B, -1, self.dims))
        patches = tf.concat([self.cls_token, patches], axis=1)
        return patches

    def call(self, x):
        x = self.patch_embedding(x)
        x = self._process_token(x)
        x += self.position_embedding(x)
        return x


class BEiTPreTrain(Model):
    def __init__(self, num_class, **kwargs):
        super().__init__(**kwargs)
        self.patch_encoder = BEiTPatchEncoder()
        self.beit_encoder = EncoderBlock()
        self.pred_head = Dense(num_class, "softmax")
