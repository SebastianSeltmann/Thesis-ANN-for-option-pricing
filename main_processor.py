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
import tensorflow as tf
from keras import backend as K
from keras.models import  load_model
from datetime import datetime
import itertools
import os
import matplotlib.pyplot as plt
from time import time

from config import (
    paths,
    required_precision,
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
    saveResultsForLatex,
    run_BS_as_well,
    vol_proxy,
    limit_windows,
    onCluster,
    collect_gradients_data
)
from models import (
    full_model,
    multitask_model,
    black_scholes_price,

    stupid_model,
    deep_model,
    adding_sample_model,
    rational_model,
    rational_model_v2,
    rational_multi_model,
    custom_model,

)
from actions import (
    get_data_package,
    run_and_store_ANN,
    get_data_for_single_stock_and_day,
    # prepare_data_for_rational_approach,
    # prepare_data_for_full_approach,
    run,
    get_input_gradients_at_point,
    extract_deltas,
    run_black_scholes,
    get_gradients,
    get_SSD,
)
if not onCluster:
    from plotting_actions import (
        vol_surface_plot,
        get_and_plot,
        boxplot_SSD_distribution,
        moving_average,
        plot_error_over_epochs,
        plot_surface_of_ANN,
        scatterplot_PAD
    )
from data import (
    windows_list,
    window_combi_count,
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

def perform_experiment():

    watches = ['model_name', 'time', 'optimizer', 'lr', 'epochs', 'features', 'activation', 'layers', 'nodes',
               'batch_normalization', 'loss', 'loss_oos', 'used_synth', 'normalize', 'dropout', 'batch_size',
               'failed', 'loss_mean', 'loss_std', 'val_loss_mean', 'val_loss_std', 'stock', 'dt_start', 'dt_middle',
               'dt_end', 'duration', 'N_train', 'N_val', 'regularizer']

    cols = {col: [] for col in watches}

    interrupt_flag = False

    msg = 'Evaluating {} different settings with {} feature combinations, in {} windows, each {} times for a total of {} runs.'
    print(msg.format(int(settings_combi_count / len(active_feature_combinations)),
                     len(active_feature_combinations),
                     window_combi_count,
                     identical_reruns,
                     len(active_feature_combinations)*window_combi_count*identical_reruns
                     ))

    # tt_start = datetime.now()
    # tt_end = datetime.now()
    # print((tt_end - tt_start).seconds, end=' a- ')
    # tt_start = datetime.now()

    i = 0
    for settings in itertools.product(*settings_list): # equivalent to a bunch of nested for-loops
        i += 1
        SSD_distribution_train = []
        SSD_distribution_val = []
        act, n, l, optimizer, include_synthetic_data, dropout_rate, normalization, batch_size, regularizer, c = settings
        used_features = full_feature_combination_list[c]
        if type(l) is tuple:
            sl, il = l
            l = sl + il

        j = 0
        for window in itertools.product(*windows_list):
            j += 1
            stock, date_tuple, rerun_id = window
            dt_start, dt_middle, dt_end = date_tuple

            print('{}.{}'.format(i,j), end=' ', flush=True)

            # We reinitialize these variables to None because they will be appended to cols
            loss_oos = last_losses_mean = last_losses_std = last_val_losses_mean =\
                last_val_losses_std = None

            pattern = 'c{}_act_{}_l{}_n{}_o_{}_bn{}_do{}_s{}_no{}_bs{}'
            model_name = pattern.format(c, act, l, n, optimizer, int(batch_normalization),
                                        int(dropout_rate*10), int(include_synthetic_data),
                                        normalization, batch_size)


            if multi_target:
                model_name = 'multit_'+model_name
                print(model_name, end=': ', flush=True)

                model = multitask_model(input_dim=len(used_features), shared_layers=sl,
                                        individual_layers=il, nodes_per_layer=n,
                                        activation=act, use_batch_normalization=batch_normalization,
                                        optimizer=optimizer)
            else:
                model_name = 'full_'+model_name
                print(model_name, end=': ', flush=True)

                model = full_model(input_dim=len(used_features), num_layers=l, nodes_per_layer=n,
                                   loss='mean_squared_error', activation=act, optimizer=optimizer,
                                   use_batch_normalization=batch_normalization,
                                   dropout_rate=dropout_rate, regularizer=regularizer)

            model.name = model_name


            # when rerun_id is 0, that means we just switched to a new stock or date, or setting
            # that is when we need to get the data again
            if rerun_id == 0:

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
                N_train = len(data_package[0][0])
                N_val = len(data_package[0][2])

            if lr is not None:
                K.set_value(model.optimizer.lr, lr)

            starting_time = datetime.now()
            starting_time_str = '{:%Y-%m-%d_%H-%M}'.format(starting_time)


            _, initial_loss, _ = run_and_store_ANN(model=model, inSample=True, model_name='i_'+model_name+'_inSample',
                              nb_epochs=separate_initial_epochs, reset='yes', columns=used_features,
                              include_synth=include_synthetic_data, normalize=normalization, batch_size=batch_size,
                                                   data_package=data_package, starting_time_str=starting_time_str)



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
                                                        data_package=data_package, starting_time_str=starting_time_str)



                _, loss_oos, _ = run_and_store_ANN(model=model, inSample=False,
                                                model_name=model_name+'_outSample', reset='reuse',
                                                columns=used_features, get_deltas=True,
                                                normalize=normalization, batch_size=batch_size,
                                                   data_package=data_package, starting_time_str=starting_time_str)


                (last_losses_mean, last_losses_std, last_val_losses_mean, last_val_losses_std) = loss_tuple



                data = data_package[0]
                X_train = data[0]
                X_val = data[2]
                SSD_train = get_SSD(model, X_train)
                SSD_val = get_SSD(model, X_val)
                SSD_distribution_train.append(SSD_train)
                SSD_distribution_val.append(SSD_val)


            model_end_time = datetime.now()

            feature_string = '_'.join(used_features)
            pos_fff = feature_string.find("_ff_ind")
            # reducing the long string of fama & french factors down
            if pos_fff > -1:
                feature_string = feature_string[0:pos_fff] + "_fff"


            cols['model_name'].append(model_name)
            cols['time'].append(datetime.now())
            cols['duration'].append(model_end_time - starting_time)
            cols['N_train'].append(N_train)
            cols['N_val'].append(N_val)

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
            cols['regularizer'].append(regularizer)

            print((model_end_time - starting_time).seconds, end=' - ')
            print(loss)


            #filename = '{}_{:%Y-%m-%d_%H-%M}.h5'.format(model_name, datetime.now())
            filename = model_name + '_' + starting_time_str + '.h5'


            model.save(os.path.join(paths['all_models'], filename))


            # if rerun_id == 0:
            #     scatterplot_PAD(model, [X_train, X_val], i)
            if collect_gradients_data:
                gradient_df_columns = ['model_name', 'time', 'sample', 'feature', 'feature_value', 'gradient']
                grad_data = {key: [] for key in gradient_df_columns}

                sampling_dict = dict(train=X_train, test=X_val)
                for sample_key, points in sampling_dict.items():

                    gradients = get_gradients(model, points)
                    for feature_iloc, feature_name in enumerate(points.columns):
                        for value, gradient in zip(points.iloc[:, feature_iloc], gradients[:, feature_iloc]):
                            grad_data['model_name'].append(model_name)
                            grad_data['time'].append(starting_time)
                            grad_data['sample'].append(sample_key)
                            grad_data['feature'].append(feature_name)
                            grad_data['feature_value'].append(value)
                            grad_data['gradient'].append(gradient)

                        # for i, var in enumerate(points.columns):
                        #     x = np.array(points.iloc[:, i])
                        #     y = gradients[:, i]

                gradients_df = pd.DataFrame(grad_data)

                if limit_windows != 'mock-testing':
                    with pd.HDFStore(paths['gradients_data']) as store:
                        try:
                            previous_gradients_df = store['gradients_data']
                            merged_gradients_df = pd.concat([gradients_df, previous_gradients_df])
                        except:
                            merged_gradients_df = gradients_df
                        store['gradients_data'] = merged_gradients_df

            if interrupt_flag:
                break

            K.clear_session()
            tf.reset_default_graph()

        if j >= 5:
            if not onCluster:
                boxplot_SSD_distribution(SSD_distribution_train, used_features, 'Training Data', model_name)
                boxplot_SSD_distribution(SSD_distribution_val, used_features, 'Validation Data', model_name)

            if saveResultsForLatex:
                SSDD_df = pd.DataFrame(SSD_distribution_val, columns=used_features)

                key = 'SSDD_df' if i == 1 else 'SSDD_df_{}'.format(i)

                with pd.HDFStore(paths['data_for_latex']) as store:
                    store[key] = SSDD_df

    results_df = pd.DataFrame(cols)

    if limit_windows != 'mock-testing':
        try:
            with pd.ExcelFile(paths['results-excel']) as reader:
                previous_results = reader.parse("RunData")
            runID = previous_results['runID'].max()+1
            results_df['runID'] = runID
            merged_results = pd.concat([results_df, previous_results])
        except:
            results_df['runID'] = 1
            merged_results = results_df

        with pd.ExcelWriter(paths['results-excel']) as writer:
            merged_results.to_excel(writer, 'RunData')
            writer.save()

    # cols = {col: [] for col in watches}

    if run_BS_as_well:
        print('Running Black Scholes Benchmark')

        BS_watches = ['stock', 'dt_start', 'dt_middle', 'dt_end', 'vol_proxy', 'MSE', 'MAE']
        BS_cols = {col: [] for col in BS_watches}

        j = 0
        for window in itertools.product(*windows_list):
            j += 1
            stock, date_tuple, rerun_id = window
            dt_start, dt_middle, dt_end = date_tuple
            data_package = get_data_package(
                model='BS',
                columns=['days', 'moneyness', 'impl_volatility', 'v60', 'r'],
                stock=stock,
                start_date=dt_start,
                end_train_start_val_date=dt_middle,
                end_val_date=dt_end
            )
            MSE, MAE = run_black_scholes(data_package, vol_proxy=vol_proxy)

            BS_cols['stock'].append(stock)
            BS_cols['dt_start'].append(dt_start)
            BS_cols['dt_middle'].append(dt_middle)
            BS_cols['dt_end'].append(dt_end)
            BS_cols['vol_proxy'].append(vol_proxy)
            BS_cols['MSE'].append(MSE)
            BS_cols['MAE'].append(MAE)

        BS_results_df = pd.DataFrame(BS_cols)

        if limit_windows != 'mock-testing':
            BS_results_df['runID'] = runID
            try:
                with pd.ExcelFile(paths['results-excel-BS']) as reader:
                    BS_previous_results = reader.parse("RunData")
                #runID = BS_previous_results['runID'].max()+1
                BS_merged_results = pd.concat([BS_results_df, BS_previous_results])
            except:
                BS_merged_results = BS_results_df

            with pd.ExcelWriter(paths['results-excel-BS']) as writer:
                BS_merged_results.to_excel(writer, 'RunData')
                writer.save()


    print('Done')
    if not onCluster:
        if not cols['failed'][-1]:
            get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='prediction')
            get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='error')
            get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='calculated_delta')
            get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='scaled_option_price')

    print('Close')

if __name__ == '__main__':
    perform_experiment()