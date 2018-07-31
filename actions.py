import pandas as pd
import numpy as np
from sklearn import preprocessing
from time import time
from datetime import datetime
import os

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from matplotlib import pyplot as plt

from config import paths, required_precision

'''
sorted_train,
train,
validate,
some_stocks,
'''
from data import (
    data as dataset,
    some_stock,
    synth
)

from models import (
    black_scholes_price,
    rational_multi_model
)

def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def get_data_for_single_stock_and_day(sorted_set, stock, daterange):
    single_stock = sorted_set.loc[(slice(None), stock),:]
    single_stock.reset_index(inplace=True) # Otherwise the full index of sorted_set remains
    single_stock.set_index(['date', 'permno'], inplace=True)
    single_stock_multiple_days = []
    days = []
    for i in daterange:
        day = single_stock.index.levels[0][i]
        days.append(single_stock.index.levels[0][i])
        single_stock_multiple_days.append(sorted_set.loc[(day, stock),:])
    single_stock_multiple_days_single_df = sorted_set.loc[(days, stock),:]
    return single_stock_multiple_days_single_df

def get_data_window(start_date='2010-01-01',
                    end_train_start_val_date='2010-06-30',
                    end_val_date='2010-12-31',
                    input_columns=['days', 'moneyness'],
                    output_columns=['scaled_option_price'],
                    stock=some_stock):
    dates = dataset.index.get_level_values(0)
    stocks = dataset.index.get_level_values(1)

    idx_stock = stocks == stock
    idx_train = (dates >= start_date) & (dates < end_train_start_val_date)
    idx_validate = (dates >= end_train_start_val_date) & (end_train_start_val_date < end_val_date)

    train = dataset.loc[idx_train & idx_stock]
    validate = dataset.loc[idx_validate & idx_stock]

    X_train = train[input_columns]
    Y_train = train[output_columns]
    X_val = validate[input_columns]
    Y_val = validate[output_columns]

    data = X_train, Y_train, X_val, Y_val

    return data

def get_data_package(model, columns, include_synth, normalize,
                     start_date='2010-01-01',
                     end_train_start_val_date='2010-06-30',
                     end_val_date='2010-12-31',
                     stock=some_stock):
    number_of_features = model.input.shape[1]._value
    more_than_one_output = type(model.output) is list
    if len(columns) != number_of_features:
        raise('mismatch between model feature count and number of columns selected in data')

    if more_than_one_output:
        output_columns = ['scaled_option_price', 'perfect_hedge_1']
    else:
        output_columns = ['scaled_option_price']
    ref_columns = ['prc', 'option_price', 'strike_price', 'prc_shifted_1', 'option_price_shifted_1']

    data = get_data_window(
        input_columns=columns,
        output_columns=output_columns,
        start_date=start_date,
        end_train_start_val_date=end_train_start_val_date,
        end_val_date=end_val_date,
        stock=some_stock)
    ref_data = get_data_window(
        input_columns=ref_columns,
        output_columns=output_columns,
        start_date=start_date,
        end_train_start_val_date=end_train_start_val_date,
        end_val_date=end_val_date,
        stock=some_stock)

    if include_synth:
        X_synth = synth.loc[:,columns]
        Y_synth = synth.loc[:,output_columns]
        X_train, Y_train, X_val, Y_val = data

        new_X_train = pd.concat([X_train, X_synth])
        new_Y_train = pd.concat([Y_train, Y_synth])


        data = new_X_train, new_Y_train, X_val, Y_val


    if normalize != 'no':

        if normalize == 'rscaler':
            scaler_X = preprocessing.RobustScaler()
            scaler_Y = preprocessing.RobustScaler()
        elif normalize == 'sscaler':
            scaler_X = preprocessing.StandardScaler()
            scaler_Y = preprocessing.StandardScaler()
        elif normalize == 'mmscaler':
            scaler_X = preprocessing.MinMaxScaler()
            scaler_Y = preprocessing.MinMaxScaler()
        else:
            raise NotImplementedError

        X_train, Y_train, X_val, Y_val = data

        np_X_train_scaled = scaler_X.fit_transform(X_train)
        np_Y_train_scaled = scaler_Y.fit_transform(Y_train)
        np_X_val_scaled = scaler_X.transform(X_val)
        np_Y_val_scaled = scaler_Y.transform(Y_val)
        X_train_scaled = pd.DataFrame(np_X_train_scaled, index=X_train.index, columns=X_train.columns)
        Y_train_scaled = pd.DataFrame(np_Y_train_scaled, index=X_train.index, columns=Y_train.columns)
        X_val_scaled = pd.DataFrame(np_X_val_scaled, index=X_val.index, columns=X_val.columns)
        Y_val_scaled = pd.DataFrame(np_Y_val_scaled, index=Y_val.index, columns=Y_val.columns)

        data = X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled
    return data, X_synth, Y_synth, ref_data, scaler_X, scaler_Y

def run_and_store_ANN(model, inSample=False, reset='yes', nb_epochs=5, data_package=None, verbose=0, model_name='custom',
                      columns=['days', 'moneyness'], get_deltas=False, include_synth=False, normalize='no',
                      batch_size=25):
    if data_package is None:
        data, X_synth, Y_synth, ref_data_tuple, scaler_X, scaler_Y = get_data_package(model, columns, include_synth, normalize)
    else:
        data, X_synth, Y_synth, ref_data_tuple, scaler_X, scaler_Y = data_package


    loss, Y_prediction, history = run(model,
                                      data=data,
                                      # sample_size = 10000,
                                      nb_epochs=nb_epochs,
                                      reset=reset,
                                      offset=0,
                                      y_offset=0,
                                      plot_prediction=False,
                                      segment_plot=False,
                                      verbose=verbose,
                                      model_name=model_name,
                                      inSample=inSample,
                                      batch_size=batch_size)
    last_loss = loss[-1]
    if history is not None:
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        last_losses = np.array(loss[-100:])
        last_losses_mean = last_losses.mean()
        last_losses_std = last_losses.std()

        last_val_losses = np.array(val_loss[-100:])
        last_val_losses_mean = last_val_losses.mean()
        last_val_losses_std = last_val_losses.std()
        loss_tuple = (last_losses_mean, last_losses_std, last_val_losses_mean, last_val_losses_std)
    else:
        loss_tuple = None

    X_train, Y_train, X_val, Y_val = data
    if inSample:
        sample = X_train.copy()
        target = Y_train.copy()
        ref_data = ref_data_tuple[0]
    else:
        sample = X_val.copy()
        target = Y_val.copy()
        ref_data = ref_data_tuple[2]

    if include_synth:
        sample.drop(index=X_synth.index, inplace=True)
        # we cannot index on Y_prediction directly
        # so we temporarily let it know about its index
        # which is identical to the index of 'target'
        df = pd.DataFrame(Y_prediction, index=target.index)
        df.drop(index=X_synth.index, inplace=True)
        Y_prediction = np.array(df[0])

        target.drop(index=Y_synth.index, inplace=True)

    if normalize != 'no':
        Y_prediction = scaler_Y.inverse_transform(Y_prediction.reshape(-1,1))

        target_inv_scaled = scaler_Y.inverse_transform(target['scaled_option_price'].values.reshape(-1,1))
        sample_inv_scaled = scaler_X.inverse_transform(sample)
        sample = pd.DataFrame(sample_inv_scaled, index=sample.index, columns=sample.columns)
        target = pd.DataFrame(target_inv_scaled, index=target.index, columns=target.columns)
        '''
        X_train = scaler_X.inverse_transform(X_train)
        Y_train = scaler_Y.inverse_transform(Y_train)
        X_val = scaler_X.inverse_transform(X_val)
        Y_val = scaler_Y.inverse_transform(Y_val)
        data = X_train, Y_train, X_val, Y_val
        '''


    if get_deltas:
        deltas = extract_deltas(model, sample, ref_data.loc[:,'strike_price'])
        sample["calculated_delta"] = deltas
    if type(Y_prediction) is list:
        sample["prediction"] = Y_prediction[0]
        sample["predicted_hedge_1"] = Y_prediction[1]
        sample["perfect_hedge_1"] = target.perfect_hedge_1
        sample["hedging_error_scaled"] = (
                (
                        ref_data.loc[:, 'prc_shifted_1'] - ref_data.loc[:, 'prc']
                ) + (
                    Y_prediction[1].flatten() *   # Super sketchy:
                    (
                            ref_data.loc[:, 'option_price_shifted_1'] - ref_data.loc[:,'option_price']
                    )
                )
            ) / ref_data.loc[:, 'prc']
    else:
        sample["prediction"] = Y_prediction
    sample["scaled_option_price"] = target.scaled_option_price
    sample["error"] = sample.scaled_option_price - sample.prediction


    global model_losses_new
    store = pd.HDFStore(paths['neural_net_output'])
    try:
        if '/model_architectures' in store.keys():
            model_architectures = store['/model_architectures']
        else:
            model_architectures = pd.DataFrame(columns=['models'])
        if not model_name.startswith('rational_'):
            model_architectures.loc[model_name] = model.to_json()
        store['/model_architectures'] = model_architectures
        if reset != 'reuse':
            lossDF = pd.DataFrame(loss, columns=[model_name])
            if '/model_losses' in store.keys():
                model_losses = store['/model_losses']
                model_losses_new = lossDF.combine_first(model_losses)
            else:
                model_losses_new = lossDF
            store['/model_losses'] = model_losses_new

        store[model_name] = sample
    finally:
        store.close()
    if len(loss)> 0:
        success = loss[-1] < required_precision
    else: success = True
    return success, last_loss, loss_tuple


class TrainValTensorBoard(TensorBoard):
    # https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure/48393723
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def run(model,
        data=None,
        nb_epochs = 1,
        reset='yes',
        offset=0,
        y_offset=0,
        plot_prediction=True,
        segment_plot=False,
        sample_size=0,
        validate_size=0,
        verbose=0,
        model_name='',
        inSample=False,
        batch_size=25
        ):

    X_train, Y_train, X_val, Y_val = data

    if sample_size > 0:
        X_train = X_train[offset:offset+sample_size]
        Y_train = Y_train[offset:offset+sample_size]
    if validate_size > 0:
        X_val = X_val[y_offset:y_offset+validate_size]
        Y_val = Y_val[y_offset:y_offset+validate_size]

    # This next step is necessary when there is more than one output
    # Tensorflow expects list of columns, not list of rows
    # by default the dataframe is cast into list of rows
    Y_train = [Y_train[x] for x in Y_train.columns]
    Y_val = [Y_val[x] for x in Y_val.columns]

    # resets = ['continue', 'reuse', 'yes', 'stop-early']

    global loss, abs_error
    loss = []
    abs_error = []
    successive_successes_needed = 5
    c = 0
    history = None
    Y_prediction = None

    global tensorboard
    now_str = '{:%Y-%m-%d_%H-%M}'.format(datetime.now())
    # tensorboard = TensorBoard(log_dir=paths['tensorboard']+now_str)
    '''
    tensorboard = TensorBoard(log_dir="logs/"+model_name+"_"+now_str, histogram_freq=0,
                              write_graph=True, write_grads=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    tensorboard = TensorBoard(histogram_freq=0,
                              write_graph=True, write_grads=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)

    # callbacks = [tensorboard]
    '''
    # callbacks = [TensorBoard(log_dir="logs/"+model_name+"_"+now_str, write_graph=True, write_images=True)]
    callbacks = [TrainValTensorBoard(write_graph=False, log_dir="logs\\{}_{}".format(model_name, now_str))]

    # validation_data = None
    validation_data = (X_val, Y_val)

    if reset == 'yes':
        model.load_weights(paths['weights'])

    if reset != 'reuse':
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=verbose, callbacks=callbacks,
                            validation_data=validation_data)
        loss = history.history['loss']
        #abs_error = history.history['mean_absolute_error']
    else:
        score = model.evaluate(X_val, Y_val, verbose=0)
        loss.append(score[0])
        #abs_error.append(score[1])

    '''
        for i in range(nb_epochs):
            score = model.evaluate(X_val, Y_val, verbose=0)
            loss.append(score[0])
            abs_error.append(score[1])
            if i % 100 == 0:
                print('{}: {}'.format(i, score))
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=0, callbacks=[tensorboard])
            if reset == 'stop-early':
                if score[1] < required_precision:
                    c += 1
                    if c >= successive_successes_needed:
                        print('Good enough after {} epochs'.format(i))
                        break
                else:
                    c = 0
    '''
    if inSample:
        prediction_input_data = X_train
        prediction_target = Y_train
    else:
        prediction_input_data = X_val
        prediction_target = Y_val
    Y_prediction = model.predict(prediction_input_data, batch_size=batch_size)
    if plot_prediction:
        alpha = 0.3
        score = model.evaluate(prediction_input_data, prediction_target, verbose=2)
        print(score)

        # global Y_prediction

        if not segment_plot:
            plt.plot(prediction_target.scaled_option_price,Y_prediction, "+", label='{:1.0f}'.format(offset/sample_size+1))
        else:
            in_the_money = X_val['moneyness'] < 0.9
            out_the_money = X_val['moneyness'] > 1.1
            at_the_money = ~in_the_money & ~out_the_money
            plt.plot(prediction_target.scaled_option_price[in_the_money],Y_prediction[in_the_money], "o", label='in the money', alpha=alpha)
            plt.plot(prediction_target.scaled_option_price[at_the_money],Y_prediction[at_the_money], "o", label='at the money', alpha=alpha)
            plt.plot(prediction_target.scaled_option_price[out_the_money],Y_prediction[out_the_money], "o", label='out the money', alpha=alpha)

        plt.show()
    return loss, Y_prediction, history


def get_input_gradients_at_point(model, point):
    outputTensor = model.output
    listOfInputs = model.inputs
    gradients = K.gradients(outputTensor, listOfInputs)

    sess = K.get_session()
    evaluated_gradients = sess.run(gradients, feed_dict={model.input: np.array([point])})
    return evaluated_gradients[0][0]


def extract_deltas(model, inputs, strikes):
    outputTensor = model.output
    listOfInputs = model.inputs
    gradients = K.gradients(outputTensor, listOfInputs)

    sess = K.get_session()
    gradients_of_individual_inputs = sess.run(gradients, feed_dict={model.input: np.array(inputs)})[0]
    moneyness_derivative = pd.Series([gradients_row[0] for gradients_row in gradients_of_individual_inputs])
    moneyness_derivative.index = strikes.index
    deltas = moneyness_derivative * strikes
    return deltas


def boxplot_SSD_distribution(SSD_distribution, features, title):
    _features = features.copy()

    SSDD_df = pd.DataFrame(SSD_distribution, columns=_features)

    contributions_df = SSDD_df.divide(SSDD_df.sum(axis=1), axis=0)

    c_df_without_moneyness = contributions_df.drop('moneyness', axis=1)
    _features.remove('moneyness')

    plt.figure()
    plt.title(title)
    plt.boxplot(c_df_without_moneyness.transpose(), labels=_features, vert=False)
    plt.savefig('plots/SSD-dist-{}.png'.format(title), bbox_inches="tight")
    plt.show()

def scatterplot_PAD(model, datasets, id):
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    axes = axes.flatten()
    for points in datasets:
        gradients = get_gradients(model, points)

        for i, var in enumerate(points.columns):
            axes[i].set_title(var)
            x = np.array(points.iloc[:, i])
            y = gradients[:, i]
            axes[i].scatter(x, y, alpha=0.1, marker='o', s=0.3)  # alpha=0.01
    for unneeded in range(i + 1, len(axes)):
        fig.delaxes(axes[unneeded])
    plt.tight_layout()

    if id > 0:
        filename = 'plots/Partial_derivatives-scatter_{}.png'.format(id)
    else:
        filename = 'plots/Partial_derivatives-scatter.png'
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def get_gradients(model, inputs):
    outputTensor = model.output
    listOfInputs = model.inputs
    gradients = K.gradients(outputTensor, listOfInputs)
    sess = K.get_session()
    gradients_of_individual_inputs = sess.run(gradients, feed_dict={model.input: np.array(inputs)})[0]
    return gradients_of_individual_inputs


def get_SSD(model, inputs):
    gradients_of_individual_inputs = get_gradients(model, inputs)
    SSD = np.square(gradients_of_individual_inputs).mean(axis=0)
    return SSD


def moving_average(a, n=3):
    '''
    moving_average(abs_error)
    moving_average(abs_error,n=100)
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_error_over_epochs():
    plt.plot(abs_error, label='abs_error')
    plt.plot(moving_average(abs_error,n=100), label='abs_error')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(loss, color='blue', lw=2)
    ax.set_yscale('log')
    ax.set_title('ANN loss over epochs on logplot')
    fig.show()


def plot_surface_of_ANN(model):
    h = 50
    w = 50
    ttm = np.array(range(0, h)) / (h/2)
    moneyness = np.array(range(0, w)) / (h/2)
    X = pd.DataFrame(columns=moneyness, index=ttm)
    for t in range(h):
        for m in range(w):
            X.iloc[t,m] = model.predict(np.array([[ttm[t], moneyness[m]]]))[0][0]

    '''
    d = {'moneyness': moneyness, 'ttm': ttm}
    X = pd.DataFrame(data=d)
    predicted_price = model.predict(X)
    '''
    data = [dict(
            type='surface',
            # x = X.columns.values,
            # y = X.index.values,
            z=X.as_matrix()
        )]
    layout = go.Layout(
        title='Pricing Surface predicted by ANN',
        autosize=False,
        width=800,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50
        ),

        scene=dict(
            xaxis=dict(title='Moneyness'),
            yaxis=dict(title='Time to Maturity'),
            zaxis=dict(title='Pricing')
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig)


def vol_surface_plot(input_data, setNames=None, variable='impl_volatility'):
    data = []

    if len(input_data) > 1:
        marker = dict(
            size=3,
            symbol='3',
            opacity=0.8
        )
    else:
        marker = dict(
            size=3,
            color=input_data[0][variable].values,
            colorscale='Viridis',
            symbol='3',
            opacity=0.8
        )

    for i, day in enumerate(input_data):
        x = day.days.values
        y = day.moneyness.values
        z = day[variable].values

        if setNames:
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=marker,
                name=setNames[i]
            )
        else:
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=marker
            )
        data.append(trace)

    layout = go.Layout(
        title=variable,
        autosize=False,
        width=900,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),

        scene=dict(
            xaxis=dict(title='Time to Maturity'),
            yaxis=dict(title='Moneyness'),
            zaxis=dict(title=variable)
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='result-plots/'+variable+'-'+setNames[0] + '.html')


def get_and_plot(setNames, variable='error'):
    data = []
    store = pd.HDFStore(paths['neural_net_output'])
    for setName in setNames:
        data.append(store[setName])
    vol_surface_plot(input_data=data, setNames=setNames, variable=variable)
    store.close()


def run_black_scholes(inSample=False, filename='BS_outSample'):
    raise NotImplementedError
    sample = get_data_for_single_stock_and_day(sorted_train, some_stock, range(-1, -71, -1))
    #validate = get_sample(sorted_train, some_stock, range(-71, -150, -1))
    validate = get_data_for_single_stock_and_day(sorted_train, some_stocks[1], range(-71, -150, -1))

    hist_impl_vol = sample.impl_volatility.mean()
    hist_r = sample.r.mean()
    if inSample:
        validate = sample
    actual_prices = validate.scaled_option_price
    predicted_prices = actual_prices.copy()
    for j in range(len(validate)):
        single = validate.iloc[j]
        prediction = black_scholes_price(single.moneyness, single.days/12, hist_r, hist_impl_vol)
        predicted_prices[j] = prediction
    plt.plot(actual_prices, predicted_prices, "+")
    plt.show()

    validate["prediction"] = predicted_prices
    validate["error"] = validate.scaled_option_price - validate.prediction

    store = pd.HDFStore(paths['neural_net_output'])
    store[filename] = validate
    store.close()
    return


def follow_rational_approach(single_stock):

    single_stock = single_stock[['days', 'moneyness', 'scaled_option_price']]

    s = 0
    days = []
    for i in range(s,s+6):
        day = single_stock.index.levels[0][i]
        days.append(single_stock.index.levels[0][i])
    test_day = days.pop()
    train = single_stock.loc[days,:]
    validate = single_stock.loc[test_day,:]


    X_train = train[['days', 'moneyness']]
    Y_train = train[['scaled_option_price']]
    X_val = validate[['days', 'moneyness']]
    Y_val = validate[['scaled_option_price']]

    data = X_train, Y_train, X_val, Y_val

    model = rational_multi_model(J=5, I=9, K=5)

    run_and_store_ANN(model=model, inSample=True, filename='rational_multi_model_inSample', nb_epochs=2000,
                      reset='yes', data=data)
    run_and_store_ANN(model=model, inSample=False, filename='rational_multi_model_outSample',
                      reset='reuse', data=data)

    get_and_plot(['rational_multi_model_inSample', 'rational_multi_model_outSample'])
    get_and_plot(['rational_multi_model_inSample', 'rational_multi_model_outSample'], variable='prediction')
    get_and_plot(['rational_multi_model_inSample'], variable='scaled_option_price')

    '''
    run_and_store_ANN(model=model, inSample=True, filename='rational_multi_model_inSample', nb_epochs=1000,
                      reset='continue', data=data)
    '''
