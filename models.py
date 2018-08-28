from scipy.stats import norm
import numpy as np
from keras.models import Sequential
from keras import layers, models, regularizers
from keras.layers import Dense, Dropout
from keras.utils import plot_model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

from config import paths, onCluster

def full_model(input_dim=2, num_layers=5, nodes_per_layer = 200, loss='mape', activation='relu', optimizer='adam',
               dropout_rate=0, use_batch_normalization=False, regularizer=None):
    model = Sequential()
    if regularizer == 'l1':
        regularizer = regularizers.l1(0.01)
    if regularizer == 'l2':
        regularizer = regularizers.l2(0.01)
    model.add(Dense(nodes_per_layer, input_dim=input_dim, kernel_initializer='RandomNormal', activation=activation,
                    kernel_regularizer=regularizer))
    for i in range(num_layers - 1):
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        if use_batch_normalization:
            model.add(layers.BatchNormalization())
        model.add(Dense(nodes_per_layer, kernel_initializer='RandomNormal', activation=activation))
    model.add(Dense(1, kernel_initializer='normal'))
    metrics = ['mse', 'mae', 'mape']
    metrics.remove(loss)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.save_weights(paths['weights'])
    if not onCluster:
        plot_model(model, to_file='model-diagrams/model-f.png')
    return model


def black_scholes_pricer(m, t, r, s, option_type='call'):
    d1 = 1/(s*(t**(1/2)))*(np.log(1/m)+(r+(s**2)/2)*t)
    d2 = d1 - s*t**(1/2)
    if option_type == 'put':
        price = norm.cdf(d1) * 1 - norm.cdf(d2) * m * np.exp(-r * t)
        delta = norm.cdf(d1)
    elif option_type == 'call':
        price = norm.cdf(-d2) * m * np.exp(-r * t) - norm.cdf(-d1) * 1
        delta = - norm.cdf(-d1)
    else:
        raise ValueError("optiontype must be 'call' or 'put'")
    return price, delta

#-----------------------------------------------------------------------------
# All models below this point are not used for the analysis in the thesis.
# They are merely leftovers from the work-in-progress.
#-----------------------------------------------------------------------------

def multitask_loss(y_true, y_pred):
    return K.mean(K.sum(K.square(y_pred - y_true)), axis=-1)


def multitask_model(input_dim=2, shared_layers=5, individual_layers=3, nodes_per_layer=200,
                    activation='relu', optimizer='adam', use_batch_normalization=False):
    options_input = layers.Input(shape=(input_dim,))
    current_layer = options_input
    def configured_dense():
        return layers.Dense(nodes_per_layer, kernel_initializer='RandomNormal', activation=activation)

    for i in range(shared_layers):
        if use_batch_normalization:
            current_layer = layers.BatchNormalization()(current_layer)
        current_layer = configured_dense()(current_layer)
    current_layer_left = current_layer
    current_layer_right = current_layer

    for i in range(individual_layers):
        if use_batch_normalization:
            current_layer_left = layers.BatchNormalization()(current_layer_left)
            current_layer_right = layers.BatchNormalization()(current_layer_right)
        current_layer_left = configured_dense()(current_layer_left)
        current_layer_right = configured_dense()(current_layer_right)

    out_left = layers.Dense(1, kernel_initializer='RandomNormal', activation=activation, name='pricer')(current_layer_left)
    out_right = layers.Dense(1, kernel_initializer='RandomNormal', activation=activation, name='hedger')(current_layer_right)

    model = models.Model(inputs=[options_input], outputs=[out_left, out_right])
    model.save_weights(paths['weights'])
    if not onCluster:
        plot_model(model, to_file='model-diagrams/model-mt.png')
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'], loss_weights=[1., 1.])
    return model


def rational_model(J=5):
    options_input = layers.Input(shape=(2,))
    y = Dense(J, activation='softplus')(options_input)
    w = Dense(J, activation='sigmoid')(options_input)
    added = layers.Multiply()([y,w])
    out = layers.Dense(1, use_bias=False)(added)
    model = models.Model(inputs=[options_input], outputs=out)
    model.save_weights(paths['weights'])
    if not onCluster:
        plot_model(model, to_file='model-diagrams/rational.png')
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def rational_model_v2(J=5, asLayer=False):

    class ConstrainedWeightSoftplus(Layer):

        def __init__(self, output_dim, **kwargs):
            self.output_dim = output_dim
            super(ConstrainedWeightSoftplus, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable for this layer.
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[-1], self.output_dim),
                                          initializer='uniform',
                                          trainable=True)
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='normal',
                                        name='bias',)
            super(ConstrainedWeightSoftplus, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            weights = K.exp(self.kernel)
            output = K.bias_add(-K.dot(x, weights), self.bias)
            return K.sigmoid(output)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)

    class ConstrainedWeightSigmoid(Layer):

        def __init__(self, output_dim, **kwargs):
            self.output_dim = output_dim
            super(ConstrainedWeightSigmoid, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable for this layer.
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[-1], self.output_dim),
                                          initializer='uniform',
                                          trainable=True)
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='normal',
                                        name='bias',)
            super(ConstrainedWeightSigmoid, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            weights = K.exp(self.kernel)
            output = K.bias_add(- K.dot(x, weights), - self.bias)
            return K.softplus(output)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)

    class ConstrainedWeightDense(Layer):

        def __init__(self, output_dim, **kwargs):
            self.output_dim = output_dim
            super(ConstrainedWeightDense, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable for this layer.
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[-1], self.output_dim),
                                          initializer='uniform',
                                          trainable=True)
            super(ConstrainedWeightDense, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            weights = K.exp(self.kernel)
            return K.dot(x,weights)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)

    input = layers.Input(shape=(2,), name='input')

    moneyness = layers.Lambda(lambda x: tf.split(x,2,axis=1)[0], name='moneyness')(input)
    maturity = layers.Lambda(lambda x: tf.split(x,2,axis=1)[1], name='maturity')(input)

    left = ConstrainedWeightSoftplus(J, name='left')(moneyness)
    right = ConstrainedWeightSigmoid(J, name='right')(maturity)
    mult = layers.Multiply(name='combine')([left,right])
    out = ConstrainedWeightDense(1, name='sum')(mult)
    model = models.Model(inputs=[input], outputs=out)
    if asLayer:
        return model
    model.save_weights(paths['weights'])
    if not onCluster:
        plot_model(model, to_file='model-diagrams/rational_v2.png')
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def rational_multi_model(I=5, K=5, J=5):
    from keras import backend
    input = layers.Input(shape=(2,), name='input')
    sigmoid = layers.Dense(K, activation='sigmoid', name='sigmoid')(input)
    left = layers.Softmax(I, name='softmax')(sigmoid)

    single_models = []
    for j in range(J):
        single_models.append(rational_model_v2(J=J, asLayer=True)(input))

    right = layers.Concatenate(name='single_models')(single_models)

    mult = layers.Multiply(name='combine')([left,right])
    out = layers.Lambda(lambda x: backend.sum(x, axis=1), output_shape=(1,), name='sum')(mult)
    model = models.Model(inputs=[input], outputs=out)

    model.save_weights(paths['weights'])
    if not onCluster:
        plot_model(model, to_file='model-diagrams/rational_multi.png')
    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    return model


def deep_model(layers=5, nodes_per_layer = 200, loss='mse', activation='relu'):
    model = Sequential()
    model.add(Dense(nodes_per_layer, input_dim=2, kernel_initializer='RandomNormal', activation=activation))
    for i in range(layers-1):
        model.add(Dense(nodes_per_layer, kernel_initializer='RandomNormal', activation=activation))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='rmsprop', loss=loss, metrics=['mae'])
    model.save_weights(paths['weights'])
    if not onCluster:
        plot_model(model, to_file='model-diagrams/model-d.png')
    return model


def adding_sample_model():
    import keras

    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

    out = keras.layers.Dense(4)(added)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    if not onCluster:
        plot_model(model, to_file='model-diagrams/adding-sample.png')


def stupid_model():
    model = Sequential()
    model.add(Dense(12, input_dim=2, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(30,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
    #model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal'))
    #model.add(Dense(1,kernel_initializer='uniform',activation='softmax'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.save_weights(paths['weights'])

    if not onCluster:
        plot_model(model, to_file='model-diagrams/model-s.png')
    return model


def custom_model(J=20, activation='relu', loss='mse'):
    options_input = layers.Input(shape=(2,), name='options_input')
    y = Dense(J, activation='softplus', name='softplus')(options_input)
    w = Dense(J, activation=activation, name=activation)(options_input)
    added = layers.Multiply()([y,w])
    out = layers.Dense(1, use_bias=False)(added)
    model = models.Model(inputs=[options_input], outputs=out)
    model.save_weights(paths['weights'])
    if not onCluster:
        plot_model(model, to_file='model-diagrams/custom.png')
    model.compile(optimizer='rmsprop', loss=loss, metrics=['mae'])
    return model

