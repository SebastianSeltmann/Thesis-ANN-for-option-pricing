import os
import itertools
from numpy.random import seed as seed_np
from tensorflow import set_random_seed as seed_tf
from time import time
from datetime import datetime
import logging
import warnings
import pandas


# ----------------------------------
# Other
# ----------------------------------
logging.getLogger("tensorflow").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore',category=pandas.io.pytables.PerformanceWarning)

# ----------------------------------
# Reproducibility
# ----------------------------------
random_seed = int(time()*10000) % 2**31
seed_np(random_seed)
seed_tf(random_seed)

# ----------------------------------
# Output for Latex
# ----------------------------------
saveResultsForLatex = True
collect_gradients_data = True

# ----------------------------------
# Data Preparation
# ----------------------------------
option_type = 'call'
start_year = 2010
end_year = 2016
annualization = 252
stock_count_to_pick = 6
do_redownload_all_data = False

overlapping_windows = True
# window_limiters = ['single', 'hyper-param-search', 'final-testing', 'no', 'mock-testing']
limit_windows = 'final-testing'
use_big_time_windows = False

fundamental_columns_to_include = [
    'permno',
    'public_date',

    'ffi49',
    #'roe',
    'roa',
    'capital_ratio',

    #'pe_op_basic',
    'pe_op_dil'
]

# ----------------------------------
# Local file paths
# ----------------------------------
if os.path.isdir('D:/'):
    rootpath = "D:\\AlgoTradingData\\"
    localpath = "D:\\Dropbox\\Studium\\Master\\Thesis\\neuralnet"
    onCluster = False

elif os.path.isdir('/scratch/roklemm/option-pricing/sebbl_upload'):
    rootpath = '/scratch/roklemm/option-pricing/sebbl_upload'
    localpath = '/scratch/roklemm/option-pricing/sebbl_upload'
    onCluster = True
else:
    rootpath = "C:\\AlgoTradingData\\"
    localpath = "C:\\Dropbox\\Dropbox\\Studium\\Master\\Thesis\\neuralnet"
    onCluster = False

paths = {}
paths['data_for_latex'] = os.path.join(rootpath,"data_for_latex.h5")

paths['options_for_ann'] = os.path.join(rootpath,"options_for_ann.h5")
paths['weights'] = os.path.join(rootpath,"weights.h5")
paths['neural_net_output'] = os.path.join(rootpath,"ANN-output.h5")
paths['model_overfit'] = os.path.join(rootpath,"overfit_model.h5")
paths['model_mape'] = os.path.join(rootpath,"mape_model.h5")
paths['model_deep'] = os.path.join(rootpath,"deep_model.h5")
paths['model_best'] = os.path.join(rootpath,"model_currently_best.h5")

paths['results-excel'] = os.path.join(localpath, 'results_excel.xlsx')
paths['results-excel-BS'] = os.path.join(localpath, 'results_excel-BS.xlsx')
paths['gradients_data'] = os.path.join(rootpath,'gradients_data.h5')
paths['all_models'] = os.path.join(rootpath, 'all_models', '{:%Y-%m-%d_%H-%M}'.format(datetime.now()))


paths['prices_raw'] = os.path.join(rootpath, "prices.h5")
paths['merged'] = os.path.join(rootpath, "merged.h5")
paths['all_options_h5'] = os.path.join(rootpath,"all_options.h5")
paths['treasury'] = os.path.join(rootpath,'3months-treasury.h5')
paths['vix'] = os.path.join(rootpath,'vix.h5')
paths['dividends'] = os.path.join(rootpath,'dividends.h5')
paths['ratios'] = os.path.join(rootpath,'ratios.h5')
paths['names'] = os.path.join(rootpath,'names.h5')
paths['sp500_permnos'] = os.path.join(rootpath, 'SP500_permnos.csv')
paths['options'] = []
for y in range(start_year, end_year): #range(1996, 2017)
    paths['options'].append(os.path.join(rootpath, "OptionsData", "rawopt_" + str(y) + "AllIndices.csv"))

if not os.path.exists(paths['all_models']):
    os.makedirs(paths['all_models'])

# ----------------------------------
# Feature Selection
# ----------------------------------
ff_dummies = ['ff_ind_{}'.format(i) for i in range(49)]
feature_combinations = {
    0: ['days', 'moneyness'],
    # 1: ['days', 'moneyness', 'vix'],
    2: ['days', 'moneyness', 'vix', 'returns', 'r', 'v60', 'v20'],
    # 3: ['days', 'moneyness', 'vix', 'returns', 'r', 'v60',      ],
    # 4: ['days', 'moneyness', 'vix', 'returns', 'r',        'v20'],
    # 5: ['days', 'moneyness', 'vix', 'returns',      'v60', 'v20'],
    # 6: ['days', 'moneyness', 'vix',            'r', 'v60', 'v20'],
    # 7: ['days', 'moneyness',        'returns', 'r', 'v60', 'v20'],
    # 8: ['days', 'moneyness', 'vix', 'returns', 'r', 'v60', 'v20'] + ff_dummies,
}

mandatory_features = ['days', 'moneyness']
optional_features = []
optional_features += ['vix', 'returns', 'r', 'v60', 'v20']
optional_features += ['roe', 'roa', 'capital_ratio']
optional_features += ['pe_op_dil']  # ,'pe_op_basic'

optional_features = ['r', 'v60', 'vix', 'returns', 'roa', 'capital_ratio', 'pe_op_dil']

full_feature_combination_list = []
include_only_single_features = False
for i in range(len(optional_features) + 1):
    if not include_only_single_features or i <= 1:
        for tuple in list(itertools.combinations(optional_features, i)):
            feature_selection = mandatory_features + list(tuple)
            full_feature_combination_list.append(feature_selection)

# active_feature_combinations = list(feature_combinations.keys())
# active_feature_combinations = list(range(len(full_feature_combination_list)))   # all possible combinations
# active_feature_combinations = [0, len(full_feature_combination_list)-1]         # every and nothing
# active_feature_combinations = [len(full_feature_combination_list) - 1]  # "full" model only

full_feature_combination_list = [mandatory_features] # Nothing
full_feature_combination_list += [mandatory_features+[feature] for feature in optional_features] # Singles
full_feature_combination_list += [mandatory_features + ['r', 'vix']] # BS-like
full_feature_combination_list += [mandatory_features + optional_features] # All
active_feature_combinations = list(range(len(full_feature_combination_list)))

# ----------------------------------
# Hyperparameters
# ----------------------------------
epochs = 250
loss_func = 'mse'
# if required_precision is not reached during initial training, the run is declared "failed", saving time
if loss_func == 'mape':
    required_precision = 10**5
elif loss_func == 'mse':
    required_precision = 0.01
elif loss_func == 'mae':
    required_precision = 0.1
else:
    raise ValueError

separate_initial_epochs = int(epochs / 10)
lr = None  # 0.0001
batch_normalization = False
multi_target = False
useEarlyStopping = False

identical_reruns = 1

activations = ['relu']  # 'tanh'
number_of_nodes = [250] # [250]
number_of_layers = [3] # [3]
optimizers = ['adam']
include_synthetic_datas = [True, False]
dropout_rates = [0.1]
batch_sizes = [500]  # 100,
normalizations = ['mmscaler']  # 'no', 'rscaler', 'sscaler',
regularizers = [None] # 'l1', 'l2', None

if limit_windows == 'mock-testing':
    epochs = 10
    separate_initial_epochs = 1
    required_precision = 100
    number_of_layers = [2]
    number_of_nodes = [25]

settings_list = [
    activations,
    number_of_nodes,
    number_of_layers,
    optimizers,
    include_synthetic_datas,
    dropout_rates,
    normalizations,
    batch_sizes,
    regularizers,
    active_feature_combinations
]

settings_combi_count = 1
for setting_options in settings_list:
    settings_combi_count *= len(setting_options)

# ----------------------------------
# Benchmark
# ----------------------------------
run_BS = 'yes'  # 'yes', 'no', only_BS'
vol_proxies = ['surface', 'hist_realized']  # , hist_implied
cd_of_quotes_to_consider_for_vol_surf = 7