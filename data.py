import pandas as pd

from config import (
    paths,
    start_year,
    end_year,
    identical_reruns,
    overlapping_windows,
    limit_windows,
    use_big_time_windows,
    stock_count_to_pick
)

with pd.HDFStore(paths['options_for_ann']) as store:
    data = store['data']
    synth = store['synthetic']
    availability_summary = store['availability_summary']


selected_stocks = list(availability_summary.index)[0:stock_count_to_pick]
some_stock = selected_stocks[0]

date_tuple_list = []
if use_big_time_windows:
    # Training
    start = '{}-01-01'.format(start_year)
    mid = '{}-07-01'.format(start_year)
    end = '{}-01-01'.format(start_year+1)
    date_tuple_list.append((start, mid, end))

    # Test
    start = '{}-01-01'.format(start_year+1)
    mid = '{}-07-01'.format(end_year-1)
    end = '{}-01-01'.format(end_year)
    date_tuple_list.append((start, mid, end))

else:
    for y in range(start_year, end_year):
        start = '{}-01-01'.format(y)
        mid = '{}-07-01'.format(y)
        end = '{}-01-01'.format(y+1)
        date_tuple_list.append((start, mid, end))

        if overlapping_windows and y+1 < end_year:
            start = '{}-07-01'.format(y)
            mid = '{}-01-01'.format(y+1)
            end = '{}-07-01'.format(y+1)
            date_tuple_list.append((start, mid, end))

if limit_windows == 'single':
    windows_list = [
        selected_stocks[0:1],
        date_tuple_list[0:1],
        list(range(identical_reruns))
    ]
elif limit_windows == 'hyper-param-search':
    windows_list = [
        selected_stocks,
        date_tuple_list[0:1],
        list(range(identical_reruns))
    ]
elif limit_windows == 'mock-testing':
    windows_list = [
        selected_stocks[0:2],
        date_tuple_list[1:4],
        list(range(identical_reruns))
    ]
elif limit_windows == 'final-testing':
    windows_list = [
        selected_stocks,
        date_tuple_list[1:],
        list(range(identical_reruns))
    ]
elif limit_windows is None or limit_windows == 'no':
    windows_list = [
        selected_stocks,
        date_tuple_list,
        list(range(identical_reruns))
    ]
else:
    raise ValueError

window_combi_count = len(windows_list[0])*len(windows_list[1])



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
