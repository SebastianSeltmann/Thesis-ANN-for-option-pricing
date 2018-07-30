import os
import itertools

from sensitive_config import quandl_key

# Path definition
if os.path.isdir('D:/'):
	rootpath = "D:\\AlgoTradingData\\"
else:
	rootpath = "C:\\AlgoTradingData\\"

paths = {}
paths['options_for_ann'] = rootpath + "options_for_ann_short.h5"
paths['weights'] = rootpath + "weights.h5"
paths['neural_net_output'] = rootpath + "ANN-output.h5"
paths['model_overfit'] = rootpath + "overfit_model.h5"
paths['model_mape'] = rootpath + "mape_model.h5"
paths['model_deep'] = rootpath + "deep_model.h5"
paths['model_best'] = rootpath + "model_currently_best.h5"
paths['tensorboard'] = rootpath + "tensorboard-logs\\"
paths['results-excel'] = 'results_excel.xlsx'
paths['all_models'] = rootpath + "all_models\\"


paths['h5 constituents & prices'] = rootpath + "Data[IDs, constituents, prices].h5"
paths['all_options_h5'] = rootpath + "all_options.h5"
paths['treasury'] = rootpath + '3months-treasury.h5'
paths['vix'] = rootpath + 'vix.h5'
paths['dividends'] = rootpath + 'dividends.h5'
paths['ratios'] = rootpath + 'ratios.h5'
paths['names'] = rootpath + 'names.h5'
paths['options'] = []
for y in range(1996, 2017):
    paths['options'].append(rootpath + "OptionsData\\rawopt_" + str(y) + "AllIndices.csv")

seed = 22
required_precision = 0.01

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

epochs = 1500
separate_initial_epochs = int(epochs / 10)
lr = None  # 0.0001
batch_normalization = False
multi_target = False

identical_reruns = 1

activations = ['relu']  # 'tanh'
number_of_nodes = [250]
number_of_layers = [5]
optimizers = ['adam']
include_synthetic_datas = [True]
dropout_rates = [0.1]
batch_sizes = [200]  # 100,
normalizations = ['mmscaler']  # 'no', 'rscaler', 'sscaler',
# active_feature_combinations = list(feature_combinations.keys())
# active_feature_combinations = list(range(len(full_feature_combination_list)))   # all possible combinations
# active_feature_combinations = [0, len(full_feature_combination_list)-1]         # every and nothing
active_feature_combinations = [len(full_feature_combination_list) - 1]  # "full" model only

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

plottype = 'scatter'

