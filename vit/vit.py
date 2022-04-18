import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, LayerNormalization
from transformer.transformer import MultiHeadAttention


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


# ToDo Docstrings
class PositionalEmbedding(Layer):
    """Learnable Positional Embedding for tokens"""

    def __init__(self, num_patches: int, dims: int):
        super().__init__()
        self.num_patches = num_patches
        self.embedding = Embedding(input_dim=num_patches + 2, output_dim=dims)

    def call(self, patches):
        B, N, D = tf.shape(patches)
        position = tf.range(0, N)
        pos_embedding = self.embedding(position)[tf.newaxis]
        pos_embedding = tf.repeat(pos_embedding, B, axis=0)
        return pos_embedding


# ToDo Docstrings
class PatchEncoder(Layer):
    def __init__(self, patch_size: int, num_patches: int, dims: int):
        super().__init__()
        self.patch_split = PatchPartion(patch_size)
        self.patch_embedding = PatchEmbedding(dims)
        self.position_embedding = PositionalEmbedding(num_patches, dims)

    def _append_class_token(self, patches):
        class_token = tf.zeros_like(patches[:, :1, :])
        patches = tf.concat([class_token, patches], axis=1)
        return patches

    def call(self, x):
        x = self.patch_split(x)
        x = self.patch_embedding(x)
        x = self._append_class_token(x)
        x += self.position_embedding(x)
        return x


class MLPBlock(Layer):
    def __init__(self, ffn_dims=2048, dims=512, dropout_rate=0.1, activation="gelu"):
        super().__init__()
        self.fc1 = Dense(units=ffn_dims, activation=activation)
        self.dropout1 = Dropout(rate=dropout_rate)
        self.fc2 = Dense(units=dims, activation=activation)
        self.dropout2 = Dropout(rate=dropout_rate)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class EncoderBlock(Layer):
    def __init__(self, num_heads, ffn_dims, dims, dropout_rate, activation="gelu"):
        super().__init__()
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.msa = MultiHeadAttention(num_heads=num_heads, dims=dims)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLPBlock(ffn_dims=ffn_dims, dims=dims, dropout_rate=dropout_rate, activation=activation)

    def call(self, x):
        x = self.layernorm1(x)
        x = self.msa(x, x, x)
        x = self.layernorm2(x)
        x = self.mlp(x)
        return x


class VisualTransformer(Model):
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
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size should be separated by patch_size."
        num_patches = (image_size**2) // (patch_size**2)
        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_encoder = PatchEncoder(patch_size=patch_size, num_patches=num_patches, dims=dims)
        self.transformer_encoder = [
            EncoderBlock(num_heads, ffn_dims, dims, dropout_rate, activation) for _ in range(num_layers)
        ]
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.mlp_head = Dense(units=num_classes, activation="softmax")

    def call(self, x):
        x = self.patch_encoder(x)
        for encoder in self.transformer_encoder:
            x = encoder(x)
        x = x[:, 0, :]
        x = self.layernorm(x)
        x = self.mlp_head(x)
        return x


# To Do : Make layers available in Graph Mode
"""
def build_vit(
    image_size: int,
    patch_size: int,
    num_classes=1000,
    num_layers=12,
    num_heads=12,
    ffn_dims=3072,
    dims=768,
    dropout_rate=0.1,
    activation="gelu",
):
    num_patches = (image_size**2) // (patch_size**2)
    x = Input([image_size, image_size, 3])
    y = PatchEncoder(patch_size=patch_size, num_patches=num_patches, dims=dims)(x)
    for _ in range(num_layers):
        y = EncoderBlock(num_heads, ffn_dims, dims, dropout_rate, activation)(y)
    y = y[:, 0, :]
    y = LayerNormalization(epsilon=1e-6)
    y = Dense(units=num_classes, activation="softmax")
    return Model(x, y)
"""
