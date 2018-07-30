'''
@Todo
    repair delta calculation
        it is currently based on 1 unit of change in moneyness (rather than stockprice)

    Justify Hyperparameter selection
        maybe try random hyper-parameter optimization?
            http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization


@ToTest (already implemented)
    Get rid of weird sudden spikes
        change activation function to something bounded? --> tanh
        increase batch size

    Test new variables:
        influence of earnings:
            pe_op_basic
            pe_op_dil

'''

import pandas as pd
import numpy as np
import json
from keras import backend as K
from keras.models import  load_model
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
from scipy.stats import kde
from openpyxl import load_workbook
from time import time

from config import (
    paths,
    required_precision,
    seed,
    settings_combi_count,
    active_feature_combinations,
    identical_reruns,
    settings_list,
    full_feature_combination_list,
    batch_normalization,
    multi_target,
    separate_initial_epochs,
    lr,
    epochs,
    plottype,
    windows_list,
    window_combi_count
)
from models import (
    stupid_model,
    deep_model,
    adding_sample_model,
    rational_model,
    rational_model_v2,
    rational_multi_model,
    custom_model,
    full_model,
    multitask_model,

    black_scholes_price
)
from actions import (
    seed_rng,
    get_data_for_single_stock_and_day,
    prepare_data_for_rational_approach,
    prepare_data_for_full_approach,
    run,
    run_and_store_ANN,
    get_input_gradients_at_point,
    extract_deltas,
    moving_average,
    plot_error_over_epochs,
    plot_surface_of_ANN,
    vol_surface_plot,
    get_and_plot,
    run_black_scholes,
    follow_rational_approach,
    get_data_package,
    get_SSD,
    get_gradients,
    boxplot_SSD_distribution
)
'''
from data import (
    sorted_train,
    train,
    validate,
    some_stock,
    single_stock
)
'''

def do_it():
    perform_experiment()


def perform_experiment():

    watches = ['model_name', 'time', 'optimizer', 'lr', 'epochs', 'features', 'activation', 'layers', 'nodes',
               'batch_normalization', 'loss', 'loss_oos', 'used_synth', 'normalize', 'dropout', 'batch_size',
               'failed', 'loss_mean', 'loss_std', 'val_loss_mean', 'val_loss_std', 'stock', 'dt_start', 'dt_middle',
               'dt_end', 'duration']
    cols = {}

    for watch in watches:
        cols[watch] = []
    '''
    cols = {
        'model_name': [],
        'time': [],
        'optimizer': [],
        'lr': [],
        'epochs': [],
        'features': [],
        'activation': [],
        'layers': [],
        'nodes': [],
        'batch_normalization': [],
        'loss': [],
        'loss_oos': [],
        'used_synth': [],
        'normalize': [],
        'dropout': [],
        'batch_size': [],

        'failed': [],
        'loss_mean': [],
        'loss_std': [],
        'val_loss_mean': [],
        'val_loss_std': []
    }
    '''
    interrupt_flag = False

    msg = 'Evaluating {} different settings with {} feature combinations, in {} windows, each {} times for a total of {} runs.'
    print(msg.format(int(settings_combi_count / len(active_feature_combinations)),
                     len(active_feature_combinations),
                     window_combi_count,
                     identical_reruns,
                     settings_combi_count*len(active_feature_combinations)*window_combi_count*identical_reruns
                     ))


    plt.figure(figsize=(6, 12))

    i = 0
    for settings in itertools.product(*settings_list): # equivalent to a bunch of nested for-loops
        i += 1
        SSD_distribution_train = []
        SSD_distribution_val = []
        act, n, l, optimizer, include_synthetic_data, dropout_rate, normalization, batch_size, c = settings
        used_features = full_feature_combination_list[c]
        if type(l) is tuple:
            sl, il = l
            l = sl + il

        j = 0
        for window in itertools.product(*windows_list):
            j += 1
            stock, date_tuple, rerun_id = window
            dt_start, dt_middle, dt_end = date_tuple

            print('{}.{}'.format(i,j+1), end=' ')

            # We reinitialize these variables to None because they will be appended to cols
            loss_oos = last_losses_mean = last_losses_std = last_val_losses_mean =\
                last_val_losses_std = None

            pattern = 'c{}_act_{}_l{}_n{}_o_{}_bn{}_do{}_s{}_no{}_bs{}'
            model_name = pattern.format(c, act, l, n, optimizer, int(batch_normalization),
                                        int(dropout_rate*10), int(include_synthetic_data),
                                        normalization, batch_size)


            if multi_target:
                model_name = 'multit_'+model_name
                print(model_name, end=': ')

                model = multitask_model(input_dim=len(used_features), shared_layers=sl,
                                        individual_layers=il, nodes_per_layer=n,
                                        activation=act, use_batch_normalization=batch_normalization,
                                        optimizer=optimizer)
            else:
                model_name = 'full_'+model_name
                print(model_name, end=': ')

                model = full_model(input_dim=len(used_features), num_layers=l, nodes_per_layer=n,
                                   loss='mean_squared_error', activation=act, optimizer=optimizer,
                                   use_batch_normalization=batch_normalization,
                                   dropout_rate=dropout_rate)

            model.name = model_name

            # when rerun_id is 0, that means we just switched to a new stock or date, or setting
            # that is when we need to get the data again
            if rerun_id == 0:
                # data_package = get_data_package(model, columns=used_features, include_synth=include_synthetic_data, normalize=normalization)
                data_package = get_data_package(
                    model=model,
                    columns=used_features,
                    include_synth=include_synthetic_data,
                    normalize=normalization,
                    stock=stock,
                    start_date=dt_start,
                    end_train_start_val_date=dt_middle,
                    end_val_date=dt_end
                )

            if lr is not None:
                K.set_value(model.optimizer.lr, lr)
            model_start_time = time()
            _, initial_loss, _ = run_and_store_ANN(model=model, inSample=True, model_name='i_'+model_name+'_inSample',
                              nb_epochs=separate_initial_epochs, reset='yes', columns=used_features,
                              include_synth=include_synthetic_data, normalize=normalization, batch_size=batch_size,
                                                   data_package=data_package)

            if initial_loss > required_precision:
                print('FAILED', end=' ')
                cols['failed'].append(int(True))
                loss = initial_loss

            else:
                cols['failed'].append(int(False))

                _, loss, loss_tuple = run_and_store_ANN(model=model, inSample=True, model_name=model_name+'_inSample',
                                            nb_epochs=epochs - separate_initial_epochs, reset='continue',
                                            columns=used_features, get_deltas=True,
                                            include_synth=include_synthetic_data,
                                            normalize=normalization, batch_size=batch_size,
                                                        data_package=data_package)

                _, loss_oos, _ = run_and_store_ANN(model=model, inSample=False,
                                                model_name=model_name+'_outSample', reset='reuse',
                                                columns=used_features, get_deltas=True,
                                                normalize=normalization, batch_size=batch_size,
                                                   data_package=data_package)


                (last_losses_mean, last_losses_std, last_val_losses_mean, last_val_losses_std) = loss_tuple

                data = data_package[0]
                X_train = data[2]
                X_val = data[2]
                SSD_train = get_SSD(model, X_train)
                SSD_val = get_SSD(model, X_val)
                SSD_distribution_train.append(SSD_train)
                SSD_distribution_val.append(SSD_val)

            model_end_time = time()

            feature_string = '_'.join(used_features)
            pos_fff = feature_string.find("_ff_ind")
            # reducing the long string of fama & french factors down
            if pos_fff > -1:
                feature_string = feature_string[0:pos_fff] + "_fff"


            cols['model_name'].append(model_name)
            cols['time'].append(datetime.now())
            cols['duration'].append(model_end_time - model_start_time)

            cols['stock'].append(stock)
            cols['dt_start'].append(dt_start)
            cols['dt_middle'].append(dt_middle)
            cols['dt_end'].append(dt_end)


            cols['loss'].append(loss)
            cols['loss_oos'].append(loss_oos)
            cols['loss_mean'].append(last_losses_mean)
            cols['loss_std'].append(last_losses_std)
            cols['val_loss_mean'].append(last_val_losses_mean)
            cols['val_loss_std'].append(last_val_losses_std)

            cols['epochs'].append(epochs)
            cols['optimizer'].append(optimizer)
            cols['lr'].append(lr)
            cols['features'].append(feature_string)
            cols['activation'].append(act)
            cols['layers'].append(l)
            cols['nodes'].append(n)
            cols['batch_normalization'].append(batch_normalization)

            cols['used_synth'].append(int(include_synthetic_data))
            cols['normalize'].append(normalization)
            cols['dropout'].append(dropout_rate)
            cols['batch_size'].append(batch_size)


            print(loss)

            filename = '{}_{:%Y-%m-%d_%H-%M}.h5'.format(model_name, datetime.now())
            model.save(paths['all_models'] + filename)

            # plt.figure()
            # plt.figure(figsize=(6, 12))
            if plottype is not None:
                for points in [X_train, X_val]:
                    gradients = get_gradients(model, points)

                    for i, var in enumerate(points.columns):
                        plt.subplot(6, 2, i + 1)
                        plt.title(var)
                        x = np.array(points.iloc[:, i])
                        y = gradients[:, i]
                        if plottype == 'density':
                            nbins = 100

                            k = kde.gaussian_kde([x, y], 0.1)
                            xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
                            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                            # Make the plot
                            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Blues)
                        elif plottype == 'scatter':
                            plt.scatter(x, y, alpha=len(X_train) * 0.1 / 600)  # alpha=0.01
                        else:
                            raise NotImplementedError

            if interrupt_flag:
                break

        plt.tight_layout()
        plt.savefig('plots/Partial_derivatives-scatter.png', bbox_inches="tight")
        plt.show()

        if identical_reruns >= 5:
            boxplot_SSD_distribution(SSD_distribution_train, used_features, 'Training Data')
            boxplot_SSD_distribution(SSD_distribution_val, used_features, 'Validation Data')



    results_df = pd.DataFrame(cols)

    try:
        xl = pd.ExcelFile(paths['results-excel'])
        previous_results = xl.parse("RunData")

        merged_results = pd.concat([results_df, previous_results])
    except:
        merged_results = results_df


    writer = pd.ExcelWriter(paths['results-excel'])
    merged_results.to_excel(writer, 'RunData')
    writer.save()
    writer.close()
    print('Done')
    if not cols['fail'][-1]:
        get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='prediction')
        get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='error')
        get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='calculated_delta')
        get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='scaled_option_price')

    print('Close')

    #get_and_plot([model_name+'_inSample'], variable='error')


if __name__ == '__main__':
    do_it()