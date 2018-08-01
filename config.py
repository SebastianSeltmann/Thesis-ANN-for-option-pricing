import os
import itertools
from numpy.random import seed as seed_np
from tensorflow import set_random_seed as seed_tf

# ----------------------------------
# Authentication
# ----------------------------------
from sensitive_config import quandl_key
assert quandl_key is not None

# ----------------------------------
# Reproducibility
# ----------------------------------
random_seed = 15
seed_np(random_seed)
seed_tf(random_seed)

# ----------------------------------
# Output for Latex
# ----------------------------------
saveResultsForLatex = True

# ----------------------------------
# Data Preparation
# ----------------------------------
start_year = 2010
end_year = 2016
annualization = 252
stock_count_to_pick = 5
do_redownload_all_data = False

overlapping_windows = True
# window_limiters = ['single', 'hyper-param-search', 'final-testing', 'no']
limit_windows = 'hyper-param-search'

fundamental_columns_to_include = [
    'permno',
    'public_date',

    'ffi49',
    'roe',
    'roa',
    'capital_ratio',

    'pe_op_basic',
    'pe_op_dil'
]

# ----------------------------------
# Local file paths
# ----------------------------------
if os.path.isdir('D:/'):
	rootpath = "D:\\AlgoTradingData\\"
else:
	rootpath = "C:\\AlgoTradingData\\"

paths = {}
paths['data_for_latex'] = rootpath + "data_for_latex.h5"

paths['options_for_ann'] = rootpath + "options_for_ann.h5"
paths['weights'] = rootpath + "weights.h5"
paths['neural_net_output'] = rootpath + "ANN-output.h5"
paths['model_overfit'] = rootpath + "overfit_model.h5"
paths['model_mape'] = rootpath + "mape_model.h5"
paths['model_deep'] = rootpath + "deep_model.h5"
paths['model_best'] = rootpath + "model_currently_best.h5"
paths['tensorboard'] = rootpath + "tensorboard-logs\\"
paths['results-excel'] = 'results_excel.xlsx'
paths['all_models'] = rootpath + "all_models\\"


paths['prices_raw'] = rootpath + "Data[IDs, constituents, prices].h5"
paths['all_options_h5'] = rootpath + "all_options.h5"
paths['treasury'] = rootpath + '3months-treasury.h5'
paths['vix'] = rootpath + 'vix.h5'
paths['dividends'] = rootpath + 'dividends.h5'
paths['ratios'] = rootpath + 'ratios.h5'
paths['names'] = rootpath + 'names.h5'
paths['options'] = []
for y in range(1996, 2017):
    paths['options'].append(rootpath + "OptionsData\\rawopt_" + str(y) + "AllIndices.csv")

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
active_feature_combinations = [len(full_feature_combination_list) - 1]  # "full" model only

# ----------------------------------
# Hyperparameters
# ----------------------------------
required_precision = 0.01 # if this is not reached during initial training, the run is declared "failed", saving time
epochs = 800
separate_initial_epochs = int(epochs / 10)
lr = None  # 0.0001
batch_normalization = False
multi_target = False

identical_reruns = 1

activations = ['relu']  # 'tanh'
number_of_nodes = [100, 250]
number_of_layers = [3, 5]
optimizers = ['adam']
include_synthetic_datas = [False, True]
dropout_rates = [0.1]
batch_sizes = [200, 500]  # 100,
normalizations = ['mmscaler']  # 'no', 'rscaler', 'sscaler',

settings_list = [
    activations,
    number_of_nodes,
    number_of_layers,
    optimizers,
    include_synthetic_datas,
    dropout_rates,
    normalizations,
    batch_sizes,
    active_feature_combinations
]

settings_combi_count = 1
for setting_options in settings_list:
    settings_combi_count *= len(setting_options)
