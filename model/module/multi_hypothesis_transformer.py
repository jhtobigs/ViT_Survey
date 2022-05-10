import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
class PreNorm(Model):
    def __init__(self,fn) -> None:
        super(PreNorm, self).__init__()
        self.layernorm = LayerNormalization(axis=1)
        self.fn = fn

    def call(self,x, training):
        x = self.layernorm(x)
        x = self.fn(x,training)
        return x

class MultiHaedAttention(Model):
    def __init__(self,dim, n_heads, drop_out=0.1) -> None:
        super().__init__()
        inner_dim = dim * n_heads
        self.dim = dim
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.attn = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim *3, use_bias=False)
    
        self.to_out = Sequential([
                nn.Dense(units=dim),
                nn.Dropout(rate=drop_out)
            ])

    def call(self,x ,training):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv,num_or_size_splits=3,axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv) # reshape

        dot = einsum('b h i d, b h j d -> b h i j', q, k) * self.dim ** -0.5 # reshape
        A = self.attn(dot)
        z = einsum('b h i j, b h j d -> b h i d',A, v)
        z = rearrange(z, 'b h n d -> b n (h d)')
        
        z = self.to_out(z, training)
        return z

class MLP(Model):
    def __init__(self,dim, mlp_dim) -> None:
        super().__init__()
        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return nn.Activation(gelu)
        self._layers = Sequential([
            nn.Dense(units=mlp_dim),
            GELU(),
            nn.Dropout(rate=0.1),
            nn.Dense(units=dim),
            nn.Dropout(rate=0.1)
            ])
    def call(self,x, training):
        return self._layers(x,training=training)
        
class Transformer(Model):
    def __init__(self, dim, depth, n_heads, mlp_dim, dropout=0.1) -> None:
        super().__init__()

        self._layers = []

        for _ in range(depth):
            self._layers.append([
            PreNorm(MultiHaedAttention(dim, n_heads)),
            PreNorm(MLP(dim,mlp_dim))
            ])
    def call(self, x, training) :
        for mha, mlp in self._layers:
            x = mha(x, training) + x
            x = mlp(x, training) + x

        return x

class MultiHypothesisTransformer(Layer):
    def __init__(self, dim=351, depth=8, n_heads=4, mlp_dim=1024, n_joints=49):
        super().__init__()
        self.dim = dim

        self.length = n_joints * 2

        self.temporal_pos_embedding = tf.Variable(initial_value=tf.random.normal([1,(self.length),self.dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1,1,self.dim]))
        
        self.transformer = Transformer(dim, n_heads=n_heads, depth=depth, mlp_dim=mlp_dim)

        self.temporal_norm = LayerNormalization(axis=1)

    def call(self,x,training=True):
        x += self.temporal_pos_embedding
        
        x = self.transformer(x,training)

        x = self.temporal_norm(x)
        return x
        