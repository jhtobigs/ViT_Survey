import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, LayerNormalization
from model.transformer import EncoderBlock
from model.vit import PatchPartition, PatchEmbedding, PositionalEmbedding


class BlockwiseMasking(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return x


class BEiTPatchEncoder(Layer):
    def __init__(self, patch_size: int, num_patches: int, dims: int):
        super().__init__()
        self.patch_masking = BlockwiseMasking()
        self.patch_split = PatchPartition(patch_size)
        self.patch_embedding = PatchEmbedding(dims)
        self.position_embedding = PositionalEmbedding(num_patches, dims)

    def _append_class_token(self, patches):
        class_token = tf.zeros_like(patches[:, :1, :])
        patches = tf.concat([class_token, patches], axis=1)
        return patches

    def call(self, x):
        mask = self.patch_masking(x)
        x = self.patch_split(x)
        x = self.patch_embedding(x)
        x = self._append_class_token(x)
        x += self.position_embedding(x)
        return x


class BEiTPreTrain(Model):
    def __init__(self, num_class, **kwargs):
        super().__init__(**kwargs)
        self.patch_encoder = BEiTPatchEncoder()
        self.beit_encoder = EncoderBlock()
        self.pred_head = Dense(num_class, "softmax")
