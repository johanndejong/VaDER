import tensorflow as tf
import numpy as np
from utils import positional_encoding, scaled_dot_product_attention

class ImputationLayer(tf.keras.layers.Layer):
    def __init__(self, A_init):
        super(ImputationLayer, self).__init__()
        self.A = self.add_weight(
            "A", shape=A_init.shape, initializer=tf.constant_initializer(A_init))
    def call(self, X, W):
        return X * W + self.A * (1.0 - W)

class RnnDecodeTransformLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, I):
        super(RnnDecodeTransformLayer, self).__init__()
        weight_init = tf.constant_initializer(np.random.standard_normal([n_hidden, I]))
        bias_init = tf.constant_initializer(np.zeros(I) + 0.1)
        self.weight = self.add_weight(
            "weight", shape=[n_hidden, I], initializer=tf.initializers.glorot_uniform())
        self.bias = self.add_weight(
            "bias", shape=[I], initializer=tf.initializers.glorot_uniform())
    def call(self, rnn_output, batch_size):
        # rnn_output = tf.transpose(rnn_output, perm=[1, 0, 2])
        # rnn_output = tf.transpose(a=tf.stack(rnn_output), perm=[1, 0, 2])
        weight = tf.tile(tf.expand_dims(self.weight, 0), [batch_size, 1, 1])
        return tf.matmul(rnn_output, weight) + self.bias

class GmmLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, K):
        super(GmmLayer, self).__init__()
        self.mu_c_unscaled = self.add_weight(
            "mu_c_unscaled", shape=[K, n_hidden], initializer=tf.initializers.glorot_uniform())
        self.sigma2_c_unscaled = self.add_weight(
            "sigma2_c_unscaled", shape=[K, n_hidden], initializer=tf.initializers.glorot_uniform())
        self.phi_c_unscaled = self.add_weight(
            "phi_c_unscaled", shape=[K], initializer=tf.initializers.glorot_uniform())
    def call(self, _):
        mu_c = self.mu_c_unscaled
        sigma2_c = tf.nn.softplus(self.sigma2_c_unscaled, name="sigma2_c")
        phi_c = tf.nn.softmax(self.phi_c_unscaled, name="phi_c")
        return mu_c, sigma2_c, phi_c

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
    ])

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.rate = rate
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)
        if self.rate > 0.0:
            self.dropout1 = tf.keras.layers.Dropout(self.rate)
            self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        if self.rate > 0.0:
            attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        if self.rate > 0.0:
            ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.rate = rate
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)
        if self.rate > 0.0:
            self.dropout1 = tf.keras.layers.Dropout(self.rate)
            self.dropout2 = tf.keras.layers.Dropout(self.rate)
            self.dropout3 = tf.keras.layers.Dropout(self.rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        if self.rate > 0.0:
            attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        if self.rate > 0.0:
            attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        if self.rate > 0.0:
            ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, D, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.rate = rate
        self.d_model = d_model
        self.num_layers = num_layers
        # as in: https://arxiv.org/abs/1711.03905
        self.embedding = tf.keras.layers.Conv1D(d_model, kernel_size=1, activation='relu', input_shape=[D, d_model])
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        if self.rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        if self.rate > 0.0:
            x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, D, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(TransformerDecoder, self).__init__()

        self.rate = rate
        self.d_model = d_model
        self.num_layers = num_layers
        # as in: https://arxiv.org/abs/1711.03905
        self.embedding = tf.keras.layers.Conv1D(d_model, kernel_size=1, activation='relu', input_shape=[D, d_model])
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        if self.rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        if self.rate > 0.0:
            x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights
