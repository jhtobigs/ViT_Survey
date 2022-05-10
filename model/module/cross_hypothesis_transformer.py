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
        x = self.fn(x, training)
        return x
    
    def triple_call(self,x_1,x_2,x_3,training):
        x_1 = self.layernorm(x_1)
        x_2 = self.layernorm(x_2)
        x_3 = self.layernorm(x_3)

        x = self.fn(x_1,x_2,x_3,training)

        return x
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
        return self._layers(x, training)

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

    def call(self,x, training):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv,num_or_size_splits=3,axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv) # reshape
        dot = einsum('b h i d, b h j d -> b h i j', q, k) * self.dim ** -0.5 # reshape
        A = self.attn(dot)
        z = einsum('b h i j, b h j d -> b h i d',A, v)
        z = rearrange(z, 'b h n d -> b n (h d)')
        z = self.to_out(z, training)
        return z

        
class Transformer(Model):
    def __init__(self, dim, depth=3, n_heads=8, mlp_dim=1024, dropout=0.1) -> None:
        super().__init__()

        self.depth = depth
        multi_dim = dim // 3
        self.mha_1 = [] 
        self.mha_2 = []
        self.mha_3 = []
        for _ in range(2):
            self.mha_1.append(PreNorm(MultiHaedAttention(multi_dim, n_heads)))
            self.mha_2.append(PreNorm(MultiHaedAttention(multi_dim, n_heads)))
            self.mha_3.append(PreNorm(MultiHaedAttention(multi_dim, n_heads)))

        self.mlp = PreNorm(MLP(dim,mlp_dim))

    def call(self,x_1,x_2,x_3, training) :

        for i in range(2):
            x_1 = x_1 + self.mha_1[i](x_1,training)
            x_2 = x_2 + self.mha_2[i](x_2,training)
            x_3 = x_3 + self.mha_3[i](x_3,training)

        x = tf.concat([x_1,x_2,x_3],axis=2)

        x = x + self.mlp(x,training)

        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2: x.shape[2]]

        return x_1,x_2,x_3

class SelfHypothesisRefinement(Model): # MH-SA 
    def __init__(self, dim=512*3, depth=8, n_heads=9, mlp_dim=1024, length=351, batch_size=16):
        super().__init__()
        self.dim = dim
        self.drop_rate = 0.2
        self.length = length

        self.temporal_pos_embedding_1 = tf.Variable(initial_value=tf.random.normal([1,self.length,self.dim // 3]))
        self.temporal_pos_embedding_2 = tf.Variable(initial_value=tf.random.normal([1,self.length,self.dim // 3]))
        self.temporal_pos_embedding_3 = tf.Variable(initial_value=tf.random.normal([1,self.length,self.dim // 3]))
        
        self.pos_drop_1 = nn.Dropout(self.drop_rate)
        self.pos_drop_2 = nn.Dropout(self.drop_rate)
        self.pos_drop_3 = nn.Dropout(self.drop_rate)

        self.mhsa = Transformer(dim=self.dim, n_heads=n_heads, depth=depth, mlp_dim=dim) # n_heads=channels

    def call(self,x_1,x_2,x_3, training=None):
        x_1 += self.temporal_pos_embedding_1
        x_2 += self.temporal_pos_embedding_2
        x_3 += self.temporal_pos_embedding_3
        # 여까지 함.
        x_1,x_2,x_3 = self.mhsa(x_1,x_2,x_3 ,training)

        return x_1,x_2,x_3
class MultiHaedCrossAttention(Model):
    def __init__(self,dim, n_heads, drop_out=0.1) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.attn = nn.Softmax()

        self.linear_q = nn.Dense(units=dim, use_bias=False)
        self.linear_k = nn.Dense(units=dim, use_bias=False)
        self.linear_v = nn.Dense(units=dim, use_bias=False)

        self.to_out = Sequential([
                nn.Dense(units=dim),
                nn.Dropout(rate=self.drop_out)
            ])

    def call(self,x_1,x_2,x_3,training):
        B, N, C = x_1.shape
        q = rearrange(self.linear_q(x_1),'b n (h d) -> b h n d',h=self.n_heads, d=C//self.n_heads)
        k = rearrange(self.linear_k(x_2),'b n (h d) -> b h n d',h=self.n_heads, d=C//self.n_heads)
        v = rearrange(self.linear_v(x_3),'b n (h d) -> b h n d',h=self.n_heads, d=C//self.n_heads)

        dot = einsum('b h i d, b h j d -> b h i j', q, k) * self.dim ** -0.5 # reshape
        A = self.attn(dot)
        z = einsum('b h i j, b h j d -> b h i d',A, v)
        z = rearrange(z, 'b h n d -> b n (h d)')
        
        z = self.to_out(z,training)
        return z
class PartTransformer(Model):
    def __init__(self, dim, depth=3, n_heads=8, mlp_dim=1024, dropout=0.1) -> None:
        super().__init__()
        multi_dim = dim // 3
        self.norm3_11 = LayerNormalization(axis=1)
        self.norm3_12 = LayerNormalization(axis=1)
        self.norm3_13 = LayerNormalization(axis=1)
        
        self.norm3_21 = LayerNormalization(axis=1)
        self.norm3_22 = LayerNormalization(axis=1)
        self.norm3_23 = LayerNormalization(axis=1)

        self.norm3_31 = LayerNormalization(axis=1)
        self.norm3_32 = LayerNormalization(axis=1)
        self.norm3_33 = LayerNormalization(axis=1)

        self.mca_1 = PreNorm(MultiHaedCrossAttention(multi_dim,n_heads))
        self.mca_2 = PreNorm(MultiHaedCrossAttention(multi_dim,n_heads))
        self.mca_3 = PreNorm(MultiHaedCrossAttention(multi_dim,n_heads))

        self.mlp = PreNorm(MLP(dim,mlp_dim))

    def call(self, x_1, x_2, x_3, training):
        x_1 = x_1 + self.mca_1.triple_call(self.norm3_11(x_2), self.norm3_12(x_3), self.norm3_13(x_1), training)
        x_2 = x_2 + self.mca_2.triple_call(self.norm3_21(x_1), self.norm3_22(x_3), self.norm3_23(x_2), training)
        x_3 = x_3 + self.mca_3.triple_call(self.norm3_31(x_1), self.norm3_32(x_2), self.norm3_33(x_3), training)

        x = tf.concat([x_1,x_2,x_3],axis=2)
        x = x + self.mlp(x, training)

        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2: x.shape[2]]

        return x_1,x_2,x_3

class CrossHypothesisInteraction(Model):
    def __init__(self, dim=512*3, depth=8, n_heads=8, mlp_dim=1024, length=351, batch_size=16):
        super().__init__()
        self.dim = dim
        self.transformer = PartTransformer(dim=self.dim, n_heads=n_heads, depth=depth, mlp_dim=dim)
        self.temporal_norm = LayerNormalization(axis=1)

    def call(self,x_1,x_2,x_3, training):
        x_1, x_2, x_3 = self.transformer(x_1, x_2, x_3, training)
        x = tf.concat([x_1,x_2,x_3],axis=2)

        x = self.temporal_norm(x)
        return x