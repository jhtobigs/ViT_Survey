import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv1D, Dense, LayerNormalization, Flatten, BatchNormalization
from tensorflow.keras import Sequential
from model.module.multi_hypothesis_transformer import MultiHypothesisTransformer
from model.module.cross_hypothesis_transformer import SelfHypothesisRefinement, CrossHypothesisInteraction

from einops import rearrange
class MHFormer(Model):
    def __init__(self,channel,n_joints,n_frames):
        super().__init__()
        self.channel = channel
        self.out_joints = n_joints

        self.norm_1 = LayerNormalization(axis=1)
        self.norm_2 = LayerNormalization(axis=1)
        self.norm_3 = LayerNormalization(axis=1)

        self.trans_auto_1 = MultiHypothesisTransformer(dim=n_frames)
        self.trans_auto_2 = MultiHypothesisTransformer(dim=n_frames)
        self.trans_auto_3 = MultiHypothesisTransformer(dim=n_frames)

        self.encoder_1 = Conv1D(filters=self.channel,kernel_size=1)
        self.encoder_2 = Conv1D(filters=self.channel,kernel_size=1)
        self.encoder_3 = Conv1D(filters=self.channel,kernel_size=1)

        self.self_hypothesis_t = SelfHypothesisRefinement(length=n_frames)
        self.cross_hypothesis_t = CrossHypothesisInteraction(length=n_frames)

        self.fcn = Sequential([
            BatchNormalization(),
            Conv1D(filters=3*self.out_joints,kernel_size=1)
        ])

    def call(self, inputs, training=None):
        B, F, J, C = inputs.shape
        x = rearrange(inputs,'b f j c -> b (j c) f')

        x_1 = x + self.trans_auto_1(self.norm_1(x),training)
        x_2 = x_1 + self.trans_auto_2(self.norm_2(x_1),training)
        x_3 = x_2 + self.trans_auto_3(self.norm_3(x_2),training)

        x_1 = rearrange(x_1,'b j f -> b f j')
        x_2 = rearrange(x_2,'b j f -> b f j')
        x_3 = rearrange(x_3,'b j f -> b f j')

        x_1 = self.encoder_1(x_1)
        x_2 = self.encoder_2(x_2)
        x_3 = self.encoder_3(x_3)

        x_1, x_2, x_3 = self.self_hypothesis_t(x_1, x_2, x_3,training)

        x = self.cross_hypothesis_t(x_1,x_2,x_3,training)
        
        x = self.fcn(x,training)
        x = rearrange(x, 'b f (j c) -> b f j c', j=self.out_joints,c=3)
        return x
# 2D generator

