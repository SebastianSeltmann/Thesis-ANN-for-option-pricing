import pandas as pd
import tensorflow as tf
from keras import backend as K
from datetime import datetime
import itertools
import os

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
    run_BS,
    vol_proxies,
    limit_windows,
    onCluster,
    collect_gradients_data,
    useEarlyStopping,
    loss_func
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

    interrupt_flag = False

    # tt_start = datetime.now()
    # tt_end = datetime.now()
    # print((tt_end - tt_start).seconds, end=' a- ')
    # tt_start = datetime.now()
    try:
        with pd.ExcelFile(paths['results-excel']) as reader:
            runID = reader.parse("RunData")['runID'].max() + 1
    except:
        runID = 1

    if run_BS != 'only_BS':

        msg = 'Evaluating {} different settings with {} feature combinations, in {} windows, each {} times for a total of {} runs.'
        print(msg.format(int(settings_combi_count / len(active_feature_combinations)),
                         len(active_feature_combinations),
                         window_combi_count,
                         identical_reruns,
                         settings_combi_count * window_combi_count * identical_reruns
                         ))

        watches = ['model_name', 'time', 'optimizer', 'lr', 'epochs', 'features', 'activation', 'layers', 'nodes',
                   'batch_normalization', 'loss', 'loss_oos', 'used_synth', 'normalize', 'dropout', 'batch_size',
                   'failed', 'loss_mean', 'loss_std', 'val_loss_mean', 'val_loss_std', 'stock', 'dt_start', 'dt_middle',
                   'dt_end', 'duration', 'N_train', 'N_val', 'regularizer', 'useEarlyStopping', 'loss_func', 'MSHE',
                   'MSHE_oos', 'MAPHE', 'MAPHE_oos']

        cols = {col: [] for col in watches}

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

                pattern = 'c{}_act{}_lf{}_l{}_n{}_o{}_bn{}_do{}_s{}_no{}_bs{}_r{}'
                model_name = pattern.format(c, act, loss_func, l, n, optimizer, int(batch_normalization),
                                            int(dropout_rate*10), int(include_synthetic_data),
                                            normalization, batch_size, regularizer)


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
                                       loss=loss_func, activation=act, optimizer=optimizer,
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

                actual_epochs = epochs
                starting_time = datetime.now()
                starting_time_str = '{:%Y-%m-%d_%H-%M}'.format(starting_time)

                annResult = run_and_store_ANN(model=model, inSample=True, model_name='i_'+model_name+'_inSample',
                                  nb_epochs=separate_initial_epochs, reset='yes', columns=used_features,
                                  include_synth=include_synthetic_data, normalize=normalization, batch_size=batch_size,
                                                       data_package=data_package, starting_time_str=starting_time_str)
                initial_hist = annResult.history
                initial_loss = annResult.last_loss


                if initial_loss > required_precision:
                    print('FAILED', end=' ')
                    cols['failed'].append(int(True))
                    loss = initial_loss
                    if useEarlyStopping:
                        actual_epochs = len(initial_hist.history['loss'])

                else:
                    cols['failed'].append(int(False))
                    annResult = run_and_store_ANN(model=model, inSample=True, model_name=model_name+'_inSample',
                                                nb_epochs=epochs - separate_initial_epochs, reset='continue',
                                                columns=used_features, get_deltas=True,
                                                include_synth=include_synthetic_data,
                                                normalize=normalization, batch_size=batch_size,
                                                            data_package=data_package, starting_time_str=starting_time_str)
                    hist, loss, loss_tuple, MSHE, MAPHE = annResult


                    annResult = run_and_store_ANN(model=model, inSample=False,
                                                    model_name=model_name+'_outSample', reset='reuse',
                                                    columns=used_features, get_deltas=True,
                                                    normalize=normalization, batch_size=batch_size,
                                                       data_package=data_package, starting_time_str=starting_time_str)
                    loss_oos = annResult.last_loss
                    MSHE_oos = annResult.MSHE
                    MAPHE_oos = annResult.MAPHE

                    if useEarlyStopping:
                        actual_epochs = len(hist.history['loss'])+len(initial_hist.history['loss'])

                    (last_losses_mean, last_losses_std, last_val_losses_mean, last_val_losses_std) = loss_tuple



                    data, _, _, _, scaler_X, scaler_Y = data_package
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
                cols['MSHE'].append(MSHE)
                cols['MSHE_oos'].append(MSHE_oos)
                cols['MAPHE'].append(MAPHE)
                cols['MAPHE_oos'].append(MAPHE_oos)

                cols['epochs'].append(actual_epochs)
                cols['optimizer'].append(optimizer)
                cols['lr'].append(lr)
                cols['features'].append(feature_string)
                cols['activation'].append(act)
                cols['layers'].append(l)
                cols['nodes'].append(n)
                cols['batch_normalization'].append(batch_normalization)
                cols['loss_func'].append(loss_func)

                cols['used_synth'].append(int(include_synthetic_data))
                cols['normalize'].append(normalization)
                cols['dropout'].append(dropout_rate)
                cols['batch_size'].append(batch_size)
                cols['regularizer'].append(regularizer)
                cols['useEarlyStopping'].append(int(useEarlyStopping))

                print((model_end_time - starting_time).seconds, end=' - ')
                print(loss)



                #filename = '{}_{:%Y-%m-%d_%H-%M}.h5'.format(model_name, datetime.now())
                filename = model_name + '_' + starting_time_str + '.h5'
                model.save(os.path.join(paths['all_models'], filename))


                if i == 1 and j == i:
                    # sample model to be particularly investigated

                    loss, Y_prediction, history = run(model,
                                                      data=data,
                                                      reset='reuse',
                                                      plot_prediction=False,
                                                      segment_plot=False,
                                                      verbose=0,
                                                      model_name=model_name,
                                                      inSample=False,
                                                      batch_size=batch_size,
                                                      starting_time_str=starting_time_str)

                    model.save(paths['sample_model'])
                    with pd.HDFStore(paths['sample_data']) as store:
                        store['X_train'] = X_train
                        store['X_test'] = X_val
                        store['Y_train'] = data[1]
                        store['Y_test'] = data[3]
                        store['Y_prediction'] = pd.Series(Y_prediction.flatten())

                featureCounts_to_record = [len(full_feature_combination_list[-1])]
                is_All_or_None_Run = len(used_features) in featureCounts_to_record
                if collect_gradients_data and is_All_or_None_Run:
                    gradient_df_columns = ['model_name', 'time', 'sample', 'feature', 'feature_value', 'gradient',
                                           'stock', 'dt_start', 'runID', 'num_features', 'moneyness']
                    
                    grad_data = {key: [] for key in gradient_df_columns}

                    sampling_dict = dict(train=X_train, test=X_val)

                    for sample_key, points in sampling_dict.items():

                        gradients = get_gradients(model, points)
                        rescaled = scaler_X.inverse_transform(points)
                        points = pd.DataFrame(rescaled, index=points.index, columns=points.columns)

                        for feature_iloc, feature_name in enumerate(points.columns):
                            iterator = zip(
                                points.iloc[:, feature_iloc],
                                gradients[:, feature_iloc],
                                points.loc[:, 'moneyness']
                            )
                            for value, gradient, moneyness in iterator:
                                grad_data['model_name'].append(model_name)
                                grad_data['time'].append(starting_time)
                                grad_data['sample'].append(sample_key)
                                grad_data['feature'].append(feature_name)
                                grad_data['feature_value'].append(value)
                                grad_data['gradient'].append(gradient)
                                grad_data['stock'].append(stock)
                                grad_data['dt_start'].append(dt_start)
                                grad_data['runID'].append(runID)
                                grad_data['num_features'].append(len(points.columns))
                                grad_data['moneyness'].append(moneyness)

                    gradients_df = pd.DataFrame(grad_data)

                    if limit_windows != 'mock-testing':

                        with pd.HDFStore(paths['gradients_data'], mode='a') as store:
                            store.append('gradients_data', gradients_df, index=False, data_columns=True)

                if interrupt_flag:
                    break

                K.clear_session()
                tf.reset_default_graph()

            if j >= 5:
                if not onCluster and len(used_features) > 4:
                    boxplot_SSD_distribution(SSD_distribution_train, used_features, 'Training Data', model_name)
                    boxplot_SSD_distribution(SSD_distribution_val, used_features, 'Validation Data', model_name)

                if saveResultsForLatex:
                    SSDD_df_train = pd.DataFrame(SSD_distribution_train, columns=used_features)
                    SSDD_df_val = pd.DataFrame(SSD_distribution_val, columns=used_features)
                    SSDD_df_train['sample'] = 'train'
                    SSDD_df_val['sample'] = 'test'
                    merged_results = pd.concat([SSDD_df_train, SSDD_df_val])
                    merged_results['runID'] = runID
                    merged_results['used_synth'] = include_synthetic_data

                    try:
                        with pd.HDFStore(paths['data_for_latex']) as store:
                            previous_results = store['SSDD_df']
                            merged_results = pd.concat([merged_results, previous_results])
                    except:
                        pass

                    with pd.HDFStore(paths['data_for_latex']) as store:
                        store['SSDD_df'] = merged_results

                    # key = 'SSDD_df' if i == 1 else 'SSDD_df_{}'.format(i)
                    # with pd.HDFStore(paths['data_for_latex']) as store:
                    #     store[key] = SSDD_df

        results_df = pd.DataFrame(cols)
        results_df['runID'] = runID

        if limit_windows != 'mock-testing':
            try:
                with pd.ExcelFile(paths['results-excel']) as reader:
                    previous_results = reader.parse("RunData")
                merged_results = pd.concat([results_df, previous_results])
            except:
                merged_results = results_df

            with pd.ExcelWriter(paths['results-excel']) as writer:
                merged_results.to_excel(writer, 'RunData')
                writer.save()

        print('ANN calculations done')
        if not onCluster:
            if not cols['failed'][-1]:
                get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='prediction')
                get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='error')
                get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='calculated_delta')
                get_and_plot([model_name+'_inSample', model_name+'_outSample'], variable='scaled_option_price')


    if run_BS in ['yes', 'only_BS']: # not 'no'
        print('Running Black Scholes Benchmark')

        BS_watches = ['stock', 'dt_start', 'dt_middle', 'dt_end', 'vol_proxy', 'MSE', 'MAE', 'MAPE', 'MSHE', 'MAPHE']
        BS_cols = {col: [] for col in BS_watches}

        i = 0
        for vol_proxy in vol_proxies:
            i += 1
            j = 0
            for window in itertools.product(*windows_list):
                j += 1

                print('{}.{}'.format(i, j), end=' ', flush=True)
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
                MSE, MAE, MAPE, MSHE, MAPHE = run_black_scholes(data_package, vol_proxy=vol_proxy)
                print(MSE)

                BS_cols['stock'].append(stock)
                BS_cols['dt_start'].append(dt_start)
                BS_cols['dt_middle'].append(dt_middle)
                BS_cols['dt_end'].append(dt_end)
                BS_cols['vol_proxy'].append(vol_proxy)
                BS_cols['MSE'].append(MSE)
                BS_cols['MAE'].append(MAE)
                BS_cols['MAPE'].append(MAPE)
                BS_cols['MSHE'].append(MSHE)
                BS_cols['MAPHE'].append(MAPHE)

        BS_results_df = pd.DataFrame(BS_cols)

        if limit_windows != 'mock-testing':
            try:
                with pd.ExcelFile(paths['results-excel-BS']) as reader:
                    BS_previous_results = reader.parse("RunData")
                if run_BS == 'only_BS':
                    BS_results_df['runID'] = BS_previous_results.runID.max()+1
                else:
                    BS_results_df['runID'] = runID
                #runID = BS_previous_results['runID'].max()+1
                BS_merged_results = pd.concat([BS_results_df, BS_previous_results])
            except:
                BS_results_df['runID'] = 1
                BS_merged_results = BS_results_df

            with pd.ExcelWriter(paths['results-excel-BS']) as writer:
                BS_merged_results.to_excel(writer, 'RunData')
                writer.save()
        print('BS done')



    print('Close')

if __name__ == '__main__':
    perform_experiment()