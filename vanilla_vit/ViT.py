import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

class PreNorm():
    def __init__(self,fn) -> None:
        self.layernorm = nn.LayerNormalization()
        self.fn = fn

    def __call__(self,x):
        x = self.fn(self.layernorm(x))
        return x

class MultiHaedAttention(Layer):
    def __init__(self,dim, n_heads, drop_out=0) -> None:
        super().__init__()
        inner_dim = dim * n_heads
        self.dim = dim
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.attn = nn.Softmax()
#         self.W_q = nn.Dense(units=dim,use_bias=False)
#         self.W_k = nn.Dense(units=dim,use_bias=False)
#         self.W_v = nn.Dense(units=dim,use_bias=False)
        self.to_qkv = nn.Dense(units=inner_dim *3, use_bias=False)
    
        self.to_out = Sequential([
                nn.Dense(units=dim),
                nn.Dropout(rate=drop_out)
            ])

    def call(self,x):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv,num_or_size_splits=3,axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv) # reshape

        dot = einsum('b h i d, b h j d -> b h i j', q, k) * self.dim ** -0.5 # reshape
        A = self.attn(dot)
        z = einsum('b h i j, b h j d -> b h i d',A, v)
        z = rearrange(z, 'b h n d -> b n (h d)')
        
        z = self.to_out(z)
        return z

class MLP(Layer):
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
        self.layers = Sequential([
            nn.Dense(units=mlp_dim),
            GELU(),
            nn.Dropout(rate=0),
            nn.Dense(units=dim),
            nn.Dropout(rate=0)
            ])
    def call(self,x):
        return self.layers(x)
        
class Transformer(Layer):
    def __init__(self, dim, depth, n_heads, mlp_dim, dropout=0.0) -> None:
        super().__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
            PreNorm(MultiHaedAttention(dim, n_heads)),
            PreNorm(MLP(dim,mlp_dim))
            ])
    def call(self,x) :
        for mha, mlp in self.layers:
            x = mha(x) + x
            x = mlp(x) + x

        return x
        

class ViT():
    def __init__(self,patch_size, image_size, dim, depth, n_heads, mlp_dim) -> None:
        self.patch_size = patch_size
        self.dim = dim
        self.image_size = image_size

        self.num_patches = image_size // patch_size
        self.patch_embedding = Sequential([nn.Dense(units=self.dim)])

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1,(self.num_patches**2)+1,self.dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1,1,self.dim]))

        self.transformer = Transformer(dim, n_heads=n_heads, depth=depth, mlp_dim=mlp_dim)
        self.mlp_haed = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=1000)
        ])
    def call(self,images):
        images = tf.image.extract_patches(images=images,
                        sizes=[1, self.patch_size, self.patch_size, 1],
                        strides=[1, self.patch_size, self.patch_size, 1],
                        rates=[1, 1, 1, 1],
                        padding='VALID')
        b,h,w,c = images.shape
        images = tf.reshape(images,[b,h*w,-1])
        
        x = self.patch_embedding(images)

        x = tf.concat([self.cls_token,x],axis=1)
        x += self.pos_embedding

        x = self.transformer(x)
        # TODO: mlp haed 
        return x

# Linear projection - patch
# Transformer
#   L-Norm
#   MHSA
#   L-Norm
#   MLP
# MLPHead
#   GELU