from turtle import pos
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Embedding
from model.utils import create_padding_lookahed_mask, create_padding_mask, make_sine_pos_encoding
from einops.layers.tensorflow import Rearrange

# Todo : Make Doctstring


class PositionalEncoding(Layer):
    def __init__(self, max_seq_len, dims=512):
        super().__init__()
        pos_enc = make_sine_pos_encoding(max_seq_len, dims)[np.newaxis]
        self.pos_enc = tf.Variable(pos_enc, trainable=False, name="pos_enc", dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_enc[:, :seq_len, :]


class TransformerEmbedding(Layer):
    def __init__(self, vocab_size, max_seq_len, dims=512, dropout_rate=0.1):
        super().__init__()
        self.dims = dims
        self.embedding = Embedding(vocab_size, dims)
        self.positional_encoding = PositionalEncoding(max_seq_len, dims)
        self.dropout = Dropout(rate=dropout_rate)

    def call(self, x, training=True):
        x = self.embedding(x)
        x *= self.dims**0.5
        x = self.positional_encoding(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(Layer):
    def __init__(self, num_heads=8, dims=512):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.head_dims = dims / num_heads
        self.fc_q = Dense(dims)
        self.fc_k = Dense(dims)
        self.fc_v = Dense(dims)
        self.fc_linear = Dense(dims)

    def _split_heads(self, x):
        x = tf.split(x, self.num_heads, axis=-1)
        x = tf.stack(x, axis=1)
        return x

    def _merge_heads(self, x):
        x = Rearrange("b h n d -> b n (h d)")(x)
        return x

    def scale_dot_product(self, q, k, v, mask=None):
        x = tf.matmul(q, k, transpose_b=True)
        x /= tf.math.sqrt(self.head_dims)
        if mask is not None:
            x += mask * -1e9
        x = tf.nn.softmax(x, axis=-1)
        x = tf.matmul(x, v)
        return x

    def call(self, q, k, v, mask=None):
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        x = self.scale_dot_product(q, k, v, mask)
        x = self._merge_heads(x)
        x = self.fc_linear(x)
        return x


class FeedFowardNetwork(Layer):
    def __init__(self, ffn_dims=2048, dims=512, activation="relu"):
        super().__init__()
        self.fc1 = Dense(units=ffn_dims, activation=activation)
        self.fc2 = Dense(units=dims, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class EncoderBlock(Layer):
    def __init__(self, ffn_dims, num_heads, dims, activation="relu", dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, dims=dims)
        self.dropout1 = Dropout(rate=dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = FeedFowardNetwork(ffn_dims=ffn_dims, dims=dims, activation=activation)
        self.dropout2 = Dropout(rate=dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x, mask, training=True):
        y = self.mha(x, x, x, mask=mask)
        y = self.dropout1(y, training=training)
        x = self.layernorm1(x + y)

        y = self.ffn(x)
        y = self.dropout2(y, training=training)
        y = self.layernorm2(x + y)
        return y


class DecoderBlock(Layer):
    def __init__(self, ffn_dims, num_heads, dims, activation="relu", dropout_rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, dims=dims)
        self.dropout1 = Dropout(rate=dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, dims=dims)
        self.dropout2 = Dropout(rate=dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = FeedFowardNetwork(ffn_dims=ffn_dims, dims=dims, activation=activation)
        self.dropout3 = Dropout(rate=dropout_rate)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

    def call(self, x, encoder_output, look_ahead_mask, mask, training=True):
        y = self.mha1(x, x, x, mask=look_ahead_mask)
        y = self.dropout1(y, training=training)
        x = self.layernorm1(x + y)

        y = self.mha2(x, encoder_output, encoder_output, mask=mask)
        y = self.dropout2(y, training=training)
        x = self.layernorm2(x + y)

        y = self.ffn(x)
        y = self.dropout3(y, training=training)
        y = self.layernorm3(x + y)
        return y


class TransformerModel(Model):
    def __init__(
        self,
        e_vocab_size: int,
        d_vocab_size: int,
        max_seq_len=10000,
        num_block=6,
        ffn_dims=2048,
        num_heads=8,
        dims=512,
        share_embedding=False,
        activation="relu",
        dropout_rate=0.1,
    ):
        super().__init__()
        self.e_embedding = TransformerEmbedding(e_vocab_size, max_seq_len, dims, dropout_rate)
        if share_embedding:
            assert (
                e_vocab_size == d_vocab_size
            ), "If Use Embedding Sharing, encoder_vocab_size == decoder_vocab_size"
            self.d_embedding = self.e_embedding
        else:
            self.d_embedding = TransformerEmbedding(d_vocab_size, max_seq_len, dims, dropout_rate)
        self.encoder = [
            EncoderBlock(ffn_dims, num_heads, dims, activation, dropout_rate) for i in range(num_block)
        ]
        self.decoder = [
            DecoderBlock(ffn_dims, num_heads, dims, activation, dropout_rate) for i in range(num_block)
        ]

    def _get_masks(self, encoder_input, decoder_input):
        encoder_mask = create_padding_mask(encoder_input)
        decoder_mask = create_padding_mask(encoder_input)
        look_ahead_mask = create_padding_lookahed_mask(decoder_input)
        return encoder_mask, decoder_mask, look_ahead_mask

    def call(self, encoder_input, decoder_input, training=True):
        encoder_mask, decoder_mask, look_ahead_mask = self._get_masks(encoder_input, decoder_input)
        e_output = self.e_embedding(encoder_input)
        d_output = self.d_embedding(decoder_input)
        for layer in self.encoder:
            e_output = layer(e_output, encoder_mask, training=training)
        for layer in self.decoder:
            d_output = layer(d_output, e_output, look_ahead_mask, decoder_mask, training=training)
        d_output = tf.matmul(d_output, self.d_embedding.embedding.weights, transpose_b=True)
        return d_output
