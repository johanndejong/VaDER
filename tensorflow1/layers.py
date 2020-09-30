import tensorflow as tf
from functools import partial

def encode(X, D, I, cell_type, n_hidden, recurrent):
    if recurrent:  # train a recurrent autoencoder
        return encode_recurrent(X, D, I, cell_type, n_hidden)
    else:  # train a non-recurrent autoencoder
        return encode_nonrecurrent(X, n_hidden)

def decode(z, D, I, cell_type, n_hidden, recurrent, output_activation):
    if recurrent:  # train a recurrent autoencoder
        return decode_recurrent(z, D, I, cell_type, n_hidden, output_activation)
    else:  # train a non-recurrent autoencoder
        return decode_nonrecurrent(z, D, n_hidden, output_activation)

def encode_recurrent(X, D, I, cell_type, n_hidden):
    if len(n_hidden) > 1:
        return encode_multilayer_recurrent(X, D, I, cell_type, n_hidden)
    else:
        return encode_monolayer_recurrent(X, D, I, cell_type, n_hidden)

def decode_recurrent(X, D, I, cell_type, n_hidden, output_activation):
    if len(n_hidden) > 1:
        return decode_multilayer_recurrent(X, D, I, cell_type, n_hidden, output_activation)
    else:
        return decode_monolayer_recurrent(X, D, I, cell_type, n_hidden, output_activation)

def encode_monolayer_recurrent(X, D, I, cell_type, n_hidden):
    n_hidden = n_hidden[0]
    X = [tf.squeeze(t, [1]) for t in tf.split(X, D, 1)]
    if cell_type == "LSTM":
        encoder = tf.nn.rnn_cell.LSTMCell(
            num_units=n_hidden + n_hidden,  # one for mu, one for sigma2
            activation=tf.nn.tanh, name="encoder", use_peepholes=True
        )
        (_, (_, hidden)) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)
    else:
        encoder = tf.nn.rnn_cell.GRUCell(
            num_units=n_hidden + n_hidden,  # one for mu, one for sigma2
            activation=tf.nn.tanh, name="encoder"
        )
        (_, hidden) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)

    mu = tf.identity(hidden[:, :n_hidden], name="mu_tilde")
    log_sigma2 = tf.identity(hidden[:, n_hidden:], name="log_sigma2_tilde")
    return mu, log_sigma2


# encode
def encode_multilayer_recurrent(X, D, I, cell_type, n_hidden):
    X = [tf.squeeze(t, [1]) for t in tf.split(X, D, 1)]
    if cell_type == "LSTM":
        encoder = tf.nn.rnn_cell.LSTMCell(
            num_units=n_hidden[0],
            activation=tf.nn.tanh, name="encoder", use_peepholes=True
        )
        (_, (_, hidden)) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)
    else:
        encoder = tf.nn.rnn_cell.GRUCell(
            num_units=n_hidden[0],
            activation=tf.nn.tanh, name="encoder"
        )
        (_, hidden) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)
    for n in n_hidden[1:-1]:
        hidden = my_dense_layer(hidden, n)
    mu = tf.identity(my_dense_layer(hidden, n_hidden[-1], activation=None), name="mu_tilde")
    log_sigma2 = tf.identity(my_dense_layer(hidden, n_hidden[-1], activation=None),
                            name="log_sigma2_tilde")
    return mu, log_sigma2


# decode
def decode_monolayer_recurrent(z, D, I, cell_type, n_hidden, output_activation):
    n_hidden = n_hidden[0]
    weight = tf.Variable(tf.truncated_normal([n_hidden, I], dtype=tf.float32), name='weight')
    bias = tf.Variable(tf.constant(0.1, shape=[I], dtype=tf.float32), name='bias')

    input = [tf.zeros((tf.shape(z)[0], I), dtype=tf.float32) for _ in range(D)]
    if cell_type == "LSTM":
        decoder = tf.nn.rnn_cell.LSTMCell(n_hidden, name="decoder", use_peepholes=True)
        (output, (_, _)) = tf.nn.static_rnn(decoder, input, initial_state=(tf.zeros(tf.shape(z)), z), dtype=tf.float32)
    else:
        decoder = tf.nn.rnn_cell.GRUCell(n_hidden, name="decoder")
        (output, _) = tf.nn.static_rnn(decoder, input, initial_state=z, dtype=tf.float32)
    output = tf.transpose(tf.stack(output), [1, 0, 2])
    weight = tf.tile(tf.expand_dims(weight, 0), [tf.shape(z)[0], 1, 1])
    output = tf.matmul(output, weight) + bias
    x_raw = tf.identity(output, name="x_raw")
    x = output_activation(x_raw, name="x_output")

    return x, x_raw

# decode
def decode_multilayer_recurrent(z, D, I, cell_type, n_hidden, output_activation):
    n_hidden = n_hidden[::-1]
    hidden = z
    for n in n_hidden[1:]:
        hidden = my_dense_layer(hidden, n)
    weight = tf.Variable(tf.truncated_normal([n_hidden[-1], I], dtype=tf.float32),
                         name='weight')
    bias = tf.Variable(tf.constant(0.1, shape=[I], dtype=tf.float32), name='bias')

    input = [tf.zeros((tf.shape(hidden)[0], I), dtype=tf.float32) for _ in range(D)]
    if cell_type == "LSTM":
        decoder = tf.nn.rnn_cell.LSTMCell(n_hidden[-1], name="decoder", use_peepholes=True)
        (output, (_, _)) = tf.nn.static_rnn(decoder, input, initial_state=(tf.zeros(tf.shape(hidden)), hidden),
                                            dtype=tf.float32)
    else:
        decoder = tf.nn.rnn_cell.GRUCell(n_hidden[-1], name="decoder")
        (output, _) = tf.nn.static_rnn(decoder, input, initial_state=hidden, dtype=tf.float32)

    output = tf.transpose(tf.stack(output), [1, 0, 2])
    weight = tf.tile(tf.expand_dims(weight, 0), [tf.shape(hidden)[0], 1, 1])
    output = tf.matmul(output, weight) + bias
    x_raw = tf.identity(output, name="x_raw")
    x = output_activation(x_raw, name="x_output")
    return x, x_raw


# encode
def encode_nonrecurrent(X, n_hidden):
    hidden = X # tf.clip_by_value(X, eps, 1 - eps)
    for n in n_hidden[:-1]:
        hidden = my_dense_layer(hidden, n)
    mu = tf.identity(my_dense_layer(hidden, n_hidden[-1], activation=None), name="mu_tilde")
    log_sigma2 = tf.identity(my_dense_layer(hidden, n_hidden[-1], activation=None), name="log_sigma2_tilde")
    return mu, log_sigma2


# decode
def decode_nonrecurrent(z, D, n_hidden, output_activation):
    hidden = z
    for n in (n_hidden[:-1])[::-1]:
        hidden = my_dense_layer(hidden, n)
    x_raw = tf.identity(my_dense_layer(hidden, D, activation=None), name="x_raw")
    x = output_activation(x_raw, name="x_output")  # the reconstructions, one for each mixture component
    return x, x_raw

my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.softplus,  # tf.nn.softplus, # tf.nn.elu,
    kernel_initializer=None  # initializer
)


