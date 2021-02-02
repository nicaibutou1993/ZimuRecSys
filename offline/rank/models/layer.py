from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *


class FMLayer(Layer):

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                                         expect to be 3 dimensions" % (len(input_shape)))
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sum_and_square = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))

        square_and_sum = tf.reduce_sum(inputs * inputs, axis=1, keepdims=True)

        score = 0.5 * tf.reduce_sum(sum_and_square - square_and_sum, axis=2)

        return score

    def compute_output_shape(self, input_shape):
        return input_shape[0] + (1,)


class AttentionSequencePoolingLayer(Layer):

    def __init__(self, att_hidden_units=(80, 40), **kwargs):
        self.att_hidden_units = att_hidden_units
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):


        self.dense1 = Dense(self.att_hidden_units[0],activation="relu")
        self.dense2 = Dense(self.att_hidden_units[1], activation="relu")
        self.dense3 = Dense(1)

        super(AttentionSequencePoolingLayer, self).build(input_shape)



    def call(self, inputs, **kwargs):
        q, k = inputs

        q = K.repeat_elements(q, K.int_shape(k)[1],axis=1)

        a = K.concatenate([q, k, q - k, q * k], axis=-1)

        a = self.dense1(a)
        a = self.dense2(a)
        a = self.dense3(a)

        a = tf.transpose(a, (0, 2, 1))

        #a_output = K.dot(a, k)
        a_output = tf.matmul(a, k)

        return a_output

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


    def get_config(self):

        config = {"att_hidden_units":self.att_hidden_units}

        base_config = super(AttentionSequencePoolingLayer, self).get_config()

        #base_config.update(config)

        return dict(list(config.items()) + list(base_config.items()))
