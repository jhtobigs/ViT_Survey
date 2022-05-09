from requests import patch
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, Embedding, LayerNormalization
from model.vit import PositionalEmbedding, EncoderBlock
from model.utils.mask_generator import MaskingGenerator
from einops.layers.tensorflow import Rearrange


class BEiTPatchEncoder(Layer):
    def __init__(self, patch_size: int, num_patches: int, dims: int):
        super().__init__()
        self.dims = dims
        self.cls_token = tf.Variable(tf.random.truncated_normal(shape=(1, 1, dims)), trainable=True)
        self.mask_token = tf.Variable(tf.random.truncated_normal(shape=(1, 1, dims)), trainable=True)
        self.patch_masking = MaskingGenerator(patch_size, num_masking_patches=int(num_patches * 0.4))
        self.patch_embedding = Conv2D(filters=dims, kernel_size=patch_size, strides=patch_size)
        self.position_embedding = PositionalEmbedding(num_patches, dims)

    def _process_token(self, patches, block_mask):
        cls_token = tf.repeat(self.cls_token, repeats=patches.shape[0], axis=0)
        patches = patches * (1 - block_mask) + (self.mask_token * block_mask)
        patches = Rearrange("b h w d -> b (h w) d", d=self.dims)(patches)
        patches = tf.concat([cls_token, patches], axis=1)
        return patches

    def call(self, x, block_mask):
        x = self.patch_embedding(x)
        x = self._process_token(x, block_mask)
        x += self.position_embedding(x)
        return x


class BEiTPreTrain(Model):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes=1000,
        num_layers=12,
        num_heads=12,
        ffn_dims=3072,
        dims=768,
        dropout_rate=0.1,
        activation="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        assert image_size % patch_size == 0, "image_size should be separated by patch_size."
        num_patches = (image_size ** 2) // (patch_size ** 2)
        self.image_size = image_size
        self.patch_size = patch_size

        self.blockwise_masking = MaskingGenerator(int(num_patches ** 0.5), int(num_patches * 0.4))
        self.patch_encoder = BEiTPatchEncoder(patch_size, num_patches, dims)
        self.beit_encoder = [
            EncoderBlock(num_heads, ffn_dims, dims, dropout_rate, activation) for _ in range(num_layers)
        ]
        self.pred_head = Dense(num_classes, "softmax")

    def indexing_mask_token(self, x, block_mask):
        block_mask = Rearrange("b h w d -> b (h w d)")(block_mask)
        block_mask = tf.where(block_mask == 1)
        x = tf.gather_nd(x, block_mask, batch_dims=0)
        return x

    def forward_feature(self, x):
        B = x.shape[0]
        block_mask = self.blockwise_masking(B)[..., tf.newaxis]
        x = self.patch_encoder(x, block_mask)
        for encoder in self.beit_encoder:
            x = encoder(x)
        return x, block_mask

    def call(self, x):
        x, block_mask = self.forward_feature(x)
        x = self.indexing_mask_token(x, block_mask)
        x = self.pred_head(x)
        return x
