import os

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
