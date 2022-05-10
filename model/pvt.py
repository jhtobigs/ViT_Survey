import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, Flatten
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow.keras.losses import kl_divergence, categorical_crossentropy
from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)


class MultiHaedAttention(Layer):
    def __init__(self,dim, n_heads, drop_out=0):
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
class PatchEmbed(Layer):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.img_size = [img_size,img_size]
        self.patch_size = [patch_size,patch_size]
        
        self.H, self.W = self.img_size[0] // self.patch_size[0], self.img_size[0] // self.patch_size[0]
        self.num_patches = self.H * self.W
        self.proj = Conv2D(embed_dim,kernel_size=patch_size,strides=patch_size)
        self.norm = LayerNormalization(axis=1)

    def call(self, x ):
        if x.shape[1] != x.shape[2] :
            x = rearrange(x,'b c h w -> b h w c')
        print(x.shape)
        B, H, W, C = x.shape
        x = self.proj(x)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[0]

        return x, (H, W)

class PyramidVisionTransformer(Model):
    def __init__(self, img_size=224):
        super().__init__()
        self.stages = [1, 2, 3, 4]
        self.init_patch_size = (16,16)
        self.embed_dims = [64, 128, 256, 512]
        self.num_heads = [1, 2, 3, 4]
        self.depth = [3,4,6,3]
        

        for i in range(len(self.stages)):
            patch_embed = nn.Dense(units=self.embed_dims[i])
            patch_norm = nn.LayerNormalization(axis=1)
            tmp_img_size = img_size if i == 0 else img_size // (2**(i+1))
            patch_size = self.init_patch_size[0] if i == 0 else 2
            H, W = tmp_img_size // patch_size, tmp_img_size // patch_size
            num_patches =  H*W if i != self.stages[-1]-1 else H*W + 1
            pos_embed = tf.Variable(initial_value=tf.random.normal([1,num_patches,self.embed_dims[i]]))
            
            transformer = Transformer(self.embed_dims[i], depth=self.depth[i], n_heads=self.num_heads[i], mlp_dim=self.embed_dims[i]*4)

            setattr(self, f'patch_embed_{i + 1}',patch_embed)
            setattr(self, f'position_embed_{i + 1}',pos_embed)
            setattr(self, f'transformer_{i + 1}', transformer)
            setattr(self, f'num_pathces_{i + 1}',num_patches)
            setattr(self, f'layernorm_{i + 1}',patch_norm)
            setattr(self, f'HW_{i+1}',[H,W])
            setattr(self, f'patch_size_{i + 1}', patch_size)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.num_pathces_1 :
            return tf.reshape(pos_embed,[1,H*W,-1])
        else:
            pos_embed = tf.reshape(pos_embed,[1,patch_embed.shape[1],patch_embed.shape[2],-1])
            pos_embed = rearrange(pos_embed,'b h w d -> b d h w')
            pos_embed = tf.image.resize(pos_embed,size=(H,W))
            return rearrange(pos_embed,'b d h w -> b (h w) d')

    def call(self, inputs):
        outs = []
        B = inputs.shape[0]

        for i in range(len(self.stages)):
            patch_embed = getattr(self,f'patch_embed_{i + 1}')
            pos_embed = getattr(self, f'position_embed_{i + 1}')
            transformer = getattr(self, f'transformer_{i + 1}')
            patch_size = getattr(self,f'patch_size_{i + 1}')

            inputs = tf.image.extract_patches(images=inputs,
                        sizes=[1, patch_size, patch_size, 1],
                        strides=[1, patch_size, patch_size, 1],
                        rates=[1, 1, 1, 1],
                        padding='VALID')

            H, W = getattr(self, f'HW_{i + 1}')
            print(inputs.shape)
            x = patch_embed(inputs)
            x = rearrange(x,'b h w d -> b (h w) d')
            if i == self.stages[-1] - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:],x, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed,x, H, W)
            print(i , x.shape , pos_embed.shape)
            x = x+pos_embed
            x = transformer(x)
            
            x = rearrange(tf.reshape(x,[B,H,W,-1]),'b h w d -> b d h w')

            outs.append(x)

        return outs