from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import Zeros
import itertools


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
        self.dense1 = Dense(self.att_hidden_units[0], activation="relu")
        self.dense2 = Dense(self.att_hidden_units[1], activation="relu")
        self.dense3 = Dense(1)

        super(AttentionSequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        q, k = inputs

        q = K.repeat_elements(q, K.int_shape(k)[1], axis=1)

        a = K.concatenate([q, k, q - k, q * k], axis=-1)

        a = self.dense1(a)
        a = self.dense2(a)
        a = self.dense3(a)

        a = tf.transpose(a, (0, 2, 1))

        # a_output = K.dot(a, k)
        a_output = tf.matmul(a, k)

        return a_output

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])

    def get_config(self):
        config = {"att_hidden_units": self.att_hidden_units}

        base_config = super(AttentionSequencePoolingLayer, self).get_config()

        # base_config.update(config)

        return dict(list(config.items()) + list(base_config.items()))


class CrossLayer(Layer):

    def __init__(self, cross_layer_num=2, **kwargs):
        self.cross_layer_num = cross_layer_num

        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]

        self.kenrels = [self.add_weight(name="kenrel" + str(i),
                                        shape=(dim, 1),
                                        trainable=True)
                        for i in range(self.cross_layer_num)]

        self.biases = [self.add_weight(name="bias" + str(i),
                                       shape=(dim, 1),
                                       initializer=Zeros(),
                                       trainable=True) for i in range(self.cross_layer_num)]

        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.expand_dims(inputs, axis=2)

        x0 = x
        xl = x

        for i in range(self.cross_layer_num):
            _xl = tf.tensordot(xl, self.kenrels[i], axes=(1, 0))

            _xl = tf.matmul(x0, _xl)

            xl = _xl + self.biases[i] + xl

        xl = Flatten()(xl)
        return xl

    def get_config(self):
        config = {"cross_layer_num": self.cross_layer_num}

        base_config = super(CrossLayer, self).get_config()

        base_config = dict(list(base_config.items()) + list(config.items()))

        return base_config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class AFMLayer(Layer):

    def __init__(self, attention_factor, **kwargs):
        self.attention_factor = attention_factor
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense1 = Dense(16, activation="relu")

        self.dense2 = Dense(1, use_bias=False)

        self.dense3 = Dense(1, use_bias=False)

        super(AFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        embeds_vec_list = inputs

        row = []
        col = []
        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = K.concatenate(row, axis=1)
        q = K.concatenate(col, axis=1)

        inner_product = p * q

        x = self.dense1(inner_product)

        x = self.dense2(x)

        a = K.softmax(x, axis=1)

        o = tf.reduce_sum(a * inner_product, axis=1)

        afm_out = self.dense3(o)

        return afm_out

    def get_config(self):
        config = {"attention_factor": self.attention_factor}

        base_config = super(AFMLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (None, 1)


class MutiHeadSelfAttention(Layer):

    def __init__(self, att_embedding_size, head_num, use_res, **kwargs):
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res

        super(MutiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.q_dense = Dense(self.att_embedding_size * self.head_num, activation="relu")
        self.k_dense = Dense(self.att_embedding_size * self.head_num, activation="relu")
        self.v_dense = Dense(self.att_embedding_size * self.head_num, activation="relu")

        if self.use_res:
            self.r_dense = Dense(self.att_embedding_size * self.head_num, activation="relu")

        super(MutiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):

        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)

        dim = K.int_shape(q)[1]
        q = K.reshape(q, (-1, dim, self.head_num, self.att_embedding_size))
        k = K.reshape(k, (-1, dim, self.head_num, self.att_embedding_size))
        v = K.reshape(v, (-1, dim, self.head_num, self.att_embedding_size))

        q = K.permute_dimensions(q, (0, 2, 1, 3))
        k = K.permute_dimensions(k, (0, 2, 1, 3))
        v = K.permute_dimensions(v, (0, 2, 1, 3))

        a = tf.matmul(q, k, transpose_b=True)

        a = K.softmax(a, axis=-1)

        o = tf.matmul(a, v)

        o = K.reshape(o, (-1, dim, self.att_embedding_size * self.head_num))

        if self.use_res:
            o += self.r_dense(inputs)

        return o

    def get_config(self):
        config = {"att_embedding_size": self.att_embedding_size,
                  "head_num": self.head_num,
                  "use_res": self.use_res}

        base_config = super(MutiHeadSelfAttention, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.att_embedding_size * self.head_num)
