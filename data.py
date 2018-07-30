import pandas as pd

from config import paths


store = pd.HDFStore(paths['options_for_ann'])
train = store['train']
validate = store['validate']
test = store['test']
single_stock = store['single']
synth = store['synthetic']
store.close()

train.days = train.days / 365
validate.days = validate.days / 365
test.days = test.days / 365
single_stock.days = single_stock.days / 365

#synth.loc[:,[c for c in synth.columns if c != 'impl_volatility']].isna().any().any()

#some_stocks = ['10107', '81774', '14542']
some_stocks = [10104, 10107, 10137, 10138, 10299, 10516, 11081, 11552, 11600, 11674]
some_stock = some_stocks[2]
# 10137 works well enough

sorted_train = train.sort_index()


'''
    [
        'days',
        'option_price',
        'impl_volatility',
        'delta',
        'strike_price',
        'prc',
        'returns',
        'v110',
        'v60',
        'v20',
        'v5',
        'r',
        'moneyness',
        'scaled_option_price',
        
        
        'prc_shifted_1',
        'option_price_shifted_1',
        'perfect_hedge_1'
        
        
        'ffi49',
        'roe',
        'roa',
        'capital_ratio'
    ]
    
    [
        'predicted_price',
        'predicted_hedge',
        'implied_delta'
'''


'''
single_stock = sorted_train.loc[(slice(None), some_stock),:]

some_day = single_stock.iloc[-1].name[0]
single_stock_and_day = sorted_train.loc[(some_day, some_stock),:]
'''