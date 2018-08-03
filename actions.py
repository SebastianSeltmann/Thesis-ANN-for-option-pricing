import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import norm
from time import time
from datetime import datetime
import os

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard


from matplotlib import pyplot as plt

from config import paths, required_precision

from data import (
    data as dataset,
    some_stock,
    synth
)

from models import black_scholes_price


def timeit(method):
    '''
    Decorator to print the runtime of a function after it finishes
    '''
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
    idx_validate = (dates >= end_train_start_val_date) & (dates < end_val_date)

    train = dataset.loc[idx_train & idx_stock]
    validate = dataset.loc[idx_validate & idx_stock]

    X_train = train[input_columns]
    Y_train = train[output_columns]
    X_val = validate[input_columns]
    Y_val = validate[output_columns]

    data = X_train, Y_train, X_val, Y_val

    return data


def get_data_package(model, columns=['days', 'moneyness'], include_synth=False, normalize='no',
                     start_date='2010-01-01',
                     end_train_start_val_date='2010-06-30',
                     end_val_date='2010-12-31',
                     stock=some_stock):
    if model == 'BS':
        output_columns = ['scaled_option_price']
    elif model == 'BS_also_hedging':
        output_columns = ['scaled_option_price', 'perfect_hedge_1']
    else:
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
        stock=stock)
    ref_data = get_data_window(
        input_columns=ref_columns,
        output_columns=output_columns,
        start_date=start_date,
        end_train_start_val_date=end_train_start_val_date,
        end_val_date=end_val_date,
        stock=stock)

    if include_synth:
        X_synth = synth.loc[:,columns]
        Y_synth = synth.loc[:,output_columns]
        X_train, Y_train, X_val, Y_val = data

        new_X_train = pd.concat([X_train, X_synth])
        new_Y_train = pd.concat([Y_train, Y_synth])


        data = new_X_train, new_Y_train, X_val, Y_val
    else:
        X_synth = Y_synth = None


    if normalize == 'no':
        scaler_X = scaler_Y = None
    else:
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
                      batch_size=25, starting_time_str=None):
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
                                      batch_size=batch_size,
                                      starting_time_str=starting_time_str)
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

    with pd.HDFStore(paths['neural_net_output']) as store:
        store_model_architecture = False
        store_model_losses_throughout_training = False
        store_primitive_performance_metrics = True
        if store_model_architecture:
            if '/model_architectures' in store.keys():
                model_architectures = store['/model_architectures']
            else:
                model_architectures = pd.DataFrame(columns=['models'])
            if not model_name.startswith('rational_'):
                model_architectures.loc[model_name] = model.to_json()
            store['/model_architectures'] = model_architectures
        if store_model_losses_throughout_training:
            if reset != 'reuse':
                lossDF = pd.DataFrame(loss, columns=[model_name])
                if '/model_losses' in store.keys():
                    model_losses = store['/model_losses']
                    model_losses_new = lossDF.combine_first(model_losses)
                else:
                    model_losses_new = lossDF
                store['/model_losses'] = model_losses_new
        if store_primitive_performance_metrics:
            store[model_name] = sample

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
        batch_size=25,
        starting_time_str=None
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
    mse = []
    successive_successes_needed = 5
    c = 0
    history = None
    Y_prediction = None

    global tensorboard
    if starting_time_str is None:
        starting_time_str = '{:%Y-%m-%d_%H-%M}'.format(datetime.now())
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
    callbacks = [TrainValTensorBoard(write_graph=False, log_dir="logs\\{}_{}".format(model_name, starting_time_str))]

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
        from plotting_actions import actual_vs_fitted_plot
        actual_vs_fitted_plot(model, prediction_input_data, prediction_target, segment_plot, X_val, Y_prediction,
                              sample_size, offset)
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


def get_gradients(model, inputs):
    outputTensor = model.output
    listOfInputs = model.inputs
    gradients = K.gradients(outputTensor, listOfInputs)
    sess = K.get_session()
    gradients_of_individual_inputs = sess.run(gradients, feed_dict={model.input: np.array(inputs)})[0]
    return gradients_of_individual_inputs


def get_SSD(model, inputs):
    from actions import get_gradients
    gradients_of_individual_inputs = get_gradients(model, inputs)
    SSD = np.square(gradients_of_individual_inputs).mean(axis=0)
    return SSD


def run_black_scholes(data_package, inSample=False, vol_proxy='hist_realized', filename='BS_outSample'):

    # data, X_synth, Y_synth, ref_data_tuple, scaler_X, scaler_Y = data_package
    data = data_package[0]
    X_train, Y_train, X_test, Y_test = data
    if inSample:
        X_test, Y_test = X_train, Y_train

    X_train.days = X_train.days / 365
    X_test.days = X_test.days / 365

    def black_scholes_pricer(m, t, r, s, optiontype='call'):
        d1 = 1 / (s * (t ** (1 / 2))) * (np.log(m) + (r + (s ** 2) / 2) * t)
        d2 = d1 - s * t ** (1 / 2)
        if optiontype == 'call':
            price = norm.cdf(d1) * m - norm.cdf(d2) * 1 * np.exp(-r * t)
            delta = norm.cdf(d1)
        elif optiontype == 'put':
            price = norm.cdf(-d2) * 1 * np.exp(-r * t) - norm.cdf(-d1) * m
            delta = - norm.cdf(-d1)
        else:
            raise ValueError("optiontype must be 'call' or 'put'")
        return price, delta

    def BS_predict(point):
        days, moneyness, hist_impl_volatility, v60, r = tuple(point)
        if vol_proxy == 'hist_implied':
            vola = hist_impl_volatility
        elif vol_proxy == 'hist_realized':
            vola = v60
        else:
            raise ValueError
        result = black_scholes_pricer(moneyness, days, r, vola)
        return pd.Series(result, index=['price', 'delta'])

    prediction = X_test.apply(BS_predict, axis=1)

    results = X_test.copy()
    results['scaled_option_price'] = Y_test['scaled_option_price']
    results['prediction'] = prediction.price
    results['delta'] = prediction.delta
    results['error'] = results.scaled_option_price - results.prediction

    # results.error.hist(bins=200)
    # plt.show()

    MAE = results.error.abs().mean()
    MSE = results.error.pow(2).mean()

    '''
    b = results.loc[results.error.abs() == results.error.abs().max()].iloc[0]
    m = b.moneyness
    t = b.days
    r = b.r
    s = b.v60
    print(t * 252)

    df['d1'] = ((1 / (X_test.v60*X_test.days.pow(1/2)))*(np.log(X_test.moneyness)+((X_test.r+X_test.v60.pow(2))*X_test.days))).values
    df['d2'] = df.d1 - (X_test.v60*X_test.days.pow(1/2)).values
    df['p'] = df.d1.apply(norm.cdf)*X_test.moneyness.values - df.d2.apply(norm.cdf)*np.exp(-X_test.r*X_test.v60).values
    '''

    return MSE, MAE
