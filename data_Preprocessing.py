import sys
import pandas as pd
import numpy as np
import datetime
import calendar
import gc

import itertools

import math

from config import (
    paths,
    start_year,
    end_year,
    do_redownload_all_data,
    fundamental_columns_to_include,
    stock_count_to_pick,
    annualization,
    onCluster,
    optional_features,
    option_type
)

if not onCluster:
    from matplotlib import pyplot as plt
    import quandl
    import wrds as wrds

    # ----------------------------------
    # Authentication
    # ----------------------------------
    from sensitive_config import quandl_key

    quandl.ApiConfig.api_key = quandl_key

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
        'scaled_option_price'
    ]
 '''

def show_largest_objects(locality='global'):
    import operator
    import sys
    if locality == 'global':
        copy = globals().copy()
    elif locality == 'local':
        copy = locals().copy()
    elif locality == 'vars':
        copy = vars().copy()
    else:
        raise ValueError
    size_dict = {}
    for key, obj in copy.items():
        size_dict[key] = sys.getsizeof(copy[key])
    sorted_locals = sorted(size_dict.items(), key=operator.itemgetter(1))
    for tuple in sorted_locals[-6:-1]:
        name, size = tuple
        print('{}: {}'.format(name, size))


def fetch_and_store_sp500(db):
    ## ---------------------- WRDS CONNECTION  ------------------------

    if db is None:
        db = wrds.Connection()

    ## ----------------- SOURCING S&P 500 CONSTITUENTS --------------------

    # source historical S&P 500 constituents
    const = db.get_table('compm', 'idxcst_his')

    # source CRSP identifiers
    crsp_id = pd.read_csv(paths['sp500_permnos'])
    crsp_id = crsp_id[crsp_id['ending'] > "1990-12-31"]
    permnos = crsp_id['PERMNO'].values

    ## ------------------ SOURCING ACCOUNTING DATA -------------------------

    # Test source of accounting data
    # gvkeys_list = gvkeys.values
    # SP500_price = db.raw_sql("Select PRCCD,  from comp.g_secd where GVKEY in (" + ", ".join(str(x) for x in gvkeys_list) + ")")

    # No permission to access through python. Check WRDS online querry


    ## ------------------- SOURCING PRICE DATA -----------------------------
    print('Loading Price Data')
    permnolist = ", ".join(str(x) for x in permnos)
    prices = db.raw_sql(
        "Select date, permno, cusip, PRC, shrout "
        "from crspa.dsf "
        "where permno in ({}) "
        "and date between '{}-01-01' and '{}-01-01'".format(permnolist, start_year, end_year)
    )
    prices_sp50 = prices

    permnos_m = prices_sp50['permno'].unique()

    # Process the price data

    for i in permnos_m:
        if i == permnos_m[0]:
            x = prices_sp50[prices_sp50['permno'] == i][['date', 'prc']].set_index('date', drop=True)
            x.columns = [i]
            prc_merge = x
        else:
            y = prices_sp50[prices_sp50['permno'] == i][['date', 'prc']].set_index('date', drop=True)
            y.columns = [i]
            prc_merge = pd.merge(prc_merge, y, how='outer', left_index=True, right_index=True)

    print('Price Data Loaded')
    ## ----------------------------- EXPORT --------------------------------

    # with pd.ExcelWriter(paths['xlsx constituents & prices']) as writer1:
    #     const.to_excel(writer1, 'Compustat_const')
    #     crsp_id.to_excel(writer1, 'CRSP_const')
    #     prc_merge.to_excel(writer1, 'Prices')
    #     writer1.save()
    #
    # prices.to_csv(paths['raw prices'], sep='\t', encoding='utf-8')

    with pd.HDFStore(paths['prices_raw']) as store:
        store['Compustat_const'] = const
        store['CRSP_const'] = crsp_id
        store['Prices_raw'] = prices
        store['Prices'] = prc_merge
    return prc_merge, crsp_id


def store_options(option_type='call'):
    with pd.HDFStore(paths['prices_raw']) as store:
        CRSP_const = store['CRSP_const']

    ## Create constituents data frame
    open(paths['all_options_h5'], 'w').close()  # delete previous HDF
    CRSP_const = CRSP_const[CRSP_const['ending'] > '1996-01-01']

    st_y = pd.to_datetime(CRSP_const['start'])
    en_y = pd.to_datetime(CRSP_const['ending'])

    for file in paths['options']:
        print(file)
        with open(file, 'r') as o:
            data = pd.read_csv(o)

        year_index = file.find('rawopt_')
        cur_y = file[year_index + 7:year_index + 7 + 4]
        idx1 = st_y <= cur_y
        idx2 = en_y >= cur_y
        idx3 = data.best_bid > 0
        idx = idx1 & idx2 & idx3
        const = CRSP_const.loc[idx, :].reset_index(drop=True)
        listO = pd.merge(data[['id', 'date', 'days', 'best_bid', 'best_offer', 'impl_volatility', 'delta', 'strike_price']],
                         const[['PERMNO']], how='inner', left_on=['id'], right_on=['PERMNO'])
        listO = data[['id', 'date', 'days', 'best_bid', 'best_offer', 'impl_volatility', 'delta', 'strike_price']]

        option_price = (listO.best_bid + listO.best_offer) / 2
        listO = pd.concat([listO, option_price], axis=1).rename(columns={0: 'option_price'})
        listO.drop(['best_bid', 'best_offer'], axis=1, inplace=True)

        listO['date'] = pd.to_datetime(listO['date'], format='%d%b%Y')
        if option_type == 'call':
            idx3 = listO['delta'] > 0
        elif option_type == 'put':
            idx3 = listO['delta'] < 0
        else:
            raise ValueError("option_type must be either 'call' or 'put'")
        listO = listO.loc[idx3, :]
        listO['strike_price'] = listO['strike_price'] / 1000
        print(listO.shape)
        with pd.HDFStore(paths['all_options_h5']) as store:
            store.append('options' + cur_y, listO, index=False, data_columns=True)


def determine_available_wrds_data(db=None):
    print('Determining available wrds data')
    if db is None:
        db = wrds.Connection()
    col_schema = []
    col_table = []
    for library in db.schema_perm:
        for table in db.list_tables(library=library):
            col_schema.append(library)
            col_table.append(table)
    df = pd.DataFrame({
        'schema': col_schema,
        'table': col_table
    })
    col_permno = []
    col_permitted = []
    for index, row in df.iterrows():
        print("{} - {}.{}".format(index, row.schema, row.table))
        query = "select * from {}.{} limit 0".format(row.schema, row.table)
        try:
            cols = db.raw_sql(query).columns
            col_permitted.append(True)
            has_permno = cols.str.contains('permno').any()
            col_permno.append(has_permno)
        except:
            col_permitted.append(False)
            col_permno.append(False)
    df['permitted'] = col_permitted
    df['has_permno'] = col_permno
    writer = pd.ExcelWriter('./wrds-libraries.xlsx')
    df.to_excel(writer, 'Sheet1')
    writer.save()


def download_vix_data_from_quandl():
    print('Downloading vix data')
    vix = quandl.get("CHRIS/CBOE_VX1") # S&P 500 Volatility Index VIX Futures
    store = pd.HDFStore(paths['vix'])
    store['vix_quandl'] = vix
    store.close()


def download_vix_data(db):
    print('Downloading vix data')
    if db is None:
        db = wrds.Connection()
    query = ("select date, vix "
             "from cboe.cboe "
             "where date > '" + str(start_year) + "0101' "
             "and date < '" + str(end_year) + "0101' "
             )
    vix = db.raw_sql(query)

    store = pd.HDFStore(paths['vix'])
    store['vix'] = vix
    store.close()


def download_treasury_data():
    print('Downloading reasury data')
    treasury = quandl.get("FRED/DGS3MO") # 3-Month Treasury Constant Maturity Rate
    store = pd.HDFStore(paths['treasury'])
    store['treasury'] = treasury
    store.close()


def download_dividends_data(db=None):
    print('Downloading dividends data')
    if db is None:
        db = wrds.Connection()

    # data1 = db.raw_sql(
    #     # Date, GVKEY, iid, company name, closing price, shares out, volume
    #     "SELECT a.datadate, a.gvkey, a.prccd, a.cshtrd, a.cshoc * a.prccd as mktcap "
    #     "FROM comp.secd AS a  "
    #     "WHERE a.datadate > '20100101' AND a.prccd > 3 AND a.cshoc * a.prccd > 1000000000 "
    #     "LIMIT 100"
    # )

    dividends = db.raw_sql(
        "SELECT permno, paydt, dclrdt, divamt "
        "FROM crspa.dse "
        "WHERE rcrddt > '" + str(start_year) + "0101' and rcrddt < '" + str(end_year) + "0101' "
    )

    store = pd.HDFStore(paths['dividends'])
    store['dividends'] = dividends
    store.close()


def download_fundamentals_data(db=None):
    print('Downloading fundamentals data:', end=' ', flush=True)
    if db is None:
        db = wrds.Connection()
    permnos = prices_raw.permno.drop_duplicates()

    query = ("select * "
             "from wrdsapps.firm_ratio "
             "where public_date > '" + str(start_year) + "0101' "
             "and public_date < '" + str(end_year) + "0101' "
             "and permno in (" + ','.join(permnos.astype(str)) + ")")
    print('Starting Query...', end='', flush=True)
    firm_ratios = db.raw_sql(query)
    print('Finished')

    store = pd.HDFStore(paths['ratios'])
    store['ratios'] = firm_ratios
    store.close()


def download_names_data(db=None):
    print('Downloading names data:', end=' ', flush=True)
    if db is None:
        db = wrds.Connection()
    permnos = prices_raw.permno.drop_duplicates()
    query = ("select permno, comnam, ticker, namedt, nameenddt "
             "from crspa.stocknames "
             "where nameenddt > '" + str(start_year) + "0101' "
             "and namedt < '" + str(end_year) + "0101' "
             "and permno in (" + ','.join(permnos.astype(str)) + ")")
    print('Starting Query...', end='', flush=True)
    names = db.raw_sql(query)
    print('Finished')

    store = pd.HDFStore(paths['names'])
    store['names'] = names
    store.close()

def recompute_optionsdata(option_type='call'):
    store_options(option_type='call')

def redownload_all_data():
    db = wrds.Connection()

    fetch_and_store_sp500(db)

    recompute_optionsdata(option_type=option_type)

    download_vix_data(db)
    download_treasury_data()
    download_dividends_data(db)
    download_fundamentals_data(db)
    download_names_data(db)

print('Loading Stock Prices', end='', flush=True)
with pd.HDFStore(paths['prices_raw']) as store:
    prices_raw = store['Prices_raw']
    prices = store['Prices']


# Data redownload is placed here, because it needs the prices_raw DataFrame

if do_redownload_all_data:
    redownload_all_data()

print(', treasury', end='', flush=True)
with pd.HDFStore(paths['treasury']) as store:
    treasury = store['treasury']

print(', vix', end='', flush=True)
with pd.HDFStore(paths['vix']) as store:
    vix = store['vix']


'''
print(', dividends', end='', flush=True)
with pd.HDFStore(paths['dividends']) as store:
    dividends = store['dividends']
'''

print(', names')
with pd.HDFStore(paths['names']) as store:
    names = store['names']

df = pd.DataFrame()
for year in range(start_year, end_year):  # range(1996, 2016)
    print('Loading options for year {}'.format(year))

    with pd.HDFStore(paths['all_options_h5']) as store:
        options_data_year = store['options' + str(year)]
    options_data_year.rename(index=str, columns={"id": "permno"}, inplace = True)
    options_data_year.set_index(['date','permno'],inplace=True)
    df = df.append(options_data_year)

print('Calculating returns and volas')
returns = prices.pct_change()




def reshape_into_series(df):
    df.index = df.index.astype('datetime64[ns]')
    df.columns = df.columns.astype(np.int64)
    ser = df.unstack().dropna()
    return ser.swaplevel()

ser_returns = reshape_into_series(returns.rolling(60).mean() * annualization)
ser_v110 = reshape_into_series(returns.rolling(110).std() * np.sqrt(annualization))
ser_v60  = reshape_into_series(returns.rolling(60).std() * np.sqrt(annualization))
ser_v20  = reshape_into_series(returns.rolling(20).std() * np.sqrt(annualization))
ser_v5   = reshape_into_series(returns.rolling(5).std() * np.sqrt(annualization))

df['hist_impl_volatility'] = df.impl_volatility.rolling(60).mean()



print('Merging with prices data')
prices_raw['permno'] = prices_raw['permno'].astype(np.int64)
prices_raw['date'] = prices_raw['date'].astype('datetime64[ns]')
prices_raw.set_index(['date','permno'],inplace=True)

prices_raw['prc_shifted_1'] = prices_raw.groupby(level=1)['prc'].shift(-1)


merged = pd.merge(df[['days', 'option_price', 'impl_volatility', 'delta', 'strike_price', 'hist_impl_volatility']],
         prices_raw[['prc', 'prc_shifted_1']], how='inner', left_index=True, right_index=True)

print('Computing Expiration Date & shifting option_price')
merged.reset_index(inplace=True)
merged['timedelta'] = merged.loc[:, 'days'].apply(datetime.timedelta)
merged['expiration_date'] = pd.to_datetime(merged['date']) + merged.loc[:, 'timedelta']
merged.drop(columns=['timedelta'], inplace=True)
merged.set_index(['date', 'permno'], inplace=True)


print('Merging with returns data')
merged['returns'] = ser_returns
merged['v110'] = ser_v110
merged['v60'] = ser_v60
merged['v20'] = ser_v20
merged['v5'] = ser_v5


print('Cleaning memory')
del(df)
del(options_data_year)
del(prices_raw)
del(ser_v5)
del(ser_v20)
del(ser_v60)
del(ser_v110)
del(ser_returns)

del(returns)
del(prices)


print('Merging with treasury and vix data')
merged.reset_index(inplace=True)
treasury.reset_index(inplace=True)

gc.collect()

merged = pd.merge(merged, treasury, left_on='date', right_on='Date', how='inner')

# merged = pd.merge(merged, vix.loc[:,'Close'].reset_index(), left_on='date', right_on='Trade Date', how='inner')
# merged.drop(['Date', 'Trade Date'], axis=1, inplace=True)
# merged.rename(index=str, columns={"Value": "r", "Close": "vix"}, inplace=True)
merged.drop(['Date'], axis=1, inplace=True)
merged.rename(index=str, columns={"Value": "r"}, inplace=True)

gc.collect()
vix['date'] = pd.to_datetime(vix['date'])
merged = pd.merge(merged, vix, left_on='date', right_on='date')# vix.columns

print(', ratios', end='', flush=True)
with pd.HDFStore(paths['ratios']) as store:
    ratios = store['ratios']
    idx1 = ratios.public_date > datetime.date(start_year, 1, 1)
    idx2 = ratios.public_date < datetime.date(end_year, 1, 1)
    ratios = ratios.loc[idx1 & idx2]

del(idx1)
del(idx2)

print('Merging with fundamentals data')
# fundamental_columns_to_include = ratios.columns
ratios_to_merge = ratios.loc[:,fundamental_columns_to_include]

del(ratios)

# dummies = pd.get_dummies(ratios_to_merge.ffi49.fillna(0).astype(int), prefix='ff_ind')
# complete_dummy_list = ['ff_ind_{}'.format(i) for i in range(49)]
# for dummy in complete_dummy_list:
#     if dummy not in dummies.columns:
#         dummies[dummy] = 0
# 
# ratios_to_merge = pd.concat([ratios_to_merge, dummies], axis=1)
# del(dummies)

def get_last_day_of_month(date):
    last_day = calendar.monthrange(date.year, date.month)[1]
    return datetime.date(date.year, date.month, last_day)

merged['month_end_date'] = merged.loc[:,'date'].apply(get_last_day_of_month)
merged = pd.merge(merged, ratios_to_merge, left_on=['permno','month_end_date'], right_on=['permno', 'public_date'],
                  how='inner')
merged.drop(columns=['month_end_date', 'public_date'], inplace=True)

del(ratios_to_merge)
del(treasury)
del(vix)

print('Merging with names data')
idx = names.groupby(['permno'])['nameenddt'].transform(max) == names['nameenddt']
last_names = names.loc[idx, ['permno', 'comnam', 'ticker']]
last_names.permno = last_names.permno.astype('category')
last_names.comnam = last_names.comnam.astype('category')
last_names.ticker = last_names.ticker.astype('category')
del(names)

merged.permno = merged.permno.astype('category')
merged = pd.merge(merged, last_names, left_on=['permno'], right_on=['permno'], how='inner')

print('Dropping NaN values: {:.2f}%'.format(merged.isna().any(axis=1).mean()*100))
merged.dropna(how='any', inplace=True)

print('Computing moneyness')
merged['moneyness'] = merged.prc / merged.strike_price
merged['scaled_option_price'] = merged.option_price / merged.strike_price
#merged['scaled_option_price_shifted_1'] = merged.option_price_shifted_1 / merged.strike_price


merged.set_index(['date', 'permno'], inplace=True)
print('merged.shape: {}'.format(merged.shape))


print('Selecting stocks with most consistent data availability')
#names = train.reset_index().loc[:, ['permno', 'comnam', 'ticker']].drop_duplicates()

def year_and_stock(index):
    date, stock = index
    return '{}-{} {}'.format(date.year, math.ceil(date.month/6)*6, stock)

counts_with_bad_index = merged.strike_price.groupby(year_and_stock).count()
index_df = pd.DataFrame(counts_with_bad_index.index.str.split().tolist(), columns=['year', 'permno'])
counts = pd.DataFrame(dict(year=index_df.year, permno=index_df.permno, count=counts_with_bad_index.values))
counts.permno = pd.to_numeric(counts.permno)

named_counts = pd.merge(counts, last_names, left_on='permno', right_on='permno')

stocks_occur_counts = named_counts.permno.value_counts()
omnipresent_stocks = stocks_occur_counts.loc[stocks_occur_counts == 2*(end_year - start_year)].index  # occurs every year
idx = named_counts.permno.isin(omnipresent_stocks)
named_counts_omnipresent = named_counts.loc[idx]

stocks_weakest_years = named_counts_omnipresent.groupby(['permno']).min()
most_consistent_stocks = list(stocks_weakest_years.sort_values(['count']).tail(stock_count_to_pick).index)
idx = named_counts.permno.isin(most_consistent_stocks)
named_counts_most_consistent = named_counts_omnipresent.loc[idx]
named_counts_most_consistent.ticker = named_counts_most_consistent.ticker.astype('str')
named_counts_most_consistent.comnam = named_counts_most_consistent.comnam.astype('str')
availability_summary = named_counts_most_consistent.groupby('permno').min()[['count', 'ticker', 'comnam']]
print(availability_summary)

if not onCluster:
    print('Plotting available data per selected stock and year')
    pivotted = named_counts_most_consistent.pivot(index='ticker', columns='year', values='count')
    plt.figure()
    fig, ax1 = plt.subplots(1,1)
    cax = ax1.imshow(pivotted, cmap='hot')
    #fig.set_xticks((pivotted.columns), list(pivotted.columns))
    ax1.set_xticks(np.arange(len(pivotted.columns)))
    ax1.set_yticks(np.arange(len(pivotted.index)))
    ax1.set_xticklabels(list(pivotted.columns), rotation=45, ha="right")
    ax1.set_yticklabels(list(pivotted.index))
    ax1.set_title('Number of available datapoints (option quotes)\nfor each window before downsampling')
    fig.colorbar(cax)
    plt.savefig('plots/availability.png', bbox_inches="tight")
    plt.show()

downsampling_n = availability_summary['count'].min()
print('Sampling each window down to equal size: {}'.format(downsampling_n))
selected_stocks = availability_summary.index
selected_stock_data = merged.loc[merged.index.get_level_values(1).isin(selected_stocks)]
del(merged)

time_windows = []
for y in range(start_year, end_year):
    start = pd.to_datetime('{}-01-01'.format(y))
    mid =  pd.to_datetime('{}-07-01'.format(y))
    end =  pd.to_datetime('{}-01-01'.format(y+1))
    time_windows.append((start, mid))
    time_windows.append((mid, end))

ranges = [list(selected_stocks), time_windows]

time_index = selected_stock_data.index.get_level_values(0)
stock_index = selected_stock_data.index.get_level_values(1)

mi = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['date', 'permno'])
downsampled_df = pd.DataFrame(columns=selected_stock_data.columns, index=mi)

for window in itertools.product(*ranges):
    stock, time_window = window
    start, end = time_window
    idx = stock_index == stock
    idx &= time_index >= start
    idx &= time_index < end
    df = selected_stock_data.loc[idx]
    sampled_df = df.sample(downsampling_n, random_state=np.random.RandomState())
    downsampled_df = downsampled_df.append(sampled_df)


print('Reloading Stock Prices & determining price at Expiration', end='', flush=True)
with pd.HDFStore(paths['prices_raw']) as store:
    prices_raw = store['Prices_raw']
prices_raw['permno'] = prices_raw['permno'].astype(np.int64)
prices_raw['date'] = prices_raw['date'].astype('datetime64[ns]')
prices_raw.set_index(['date','permno'],inplace=True)
dateindex = prices_raw.index.get_level_values(0)
stockindex = prices_raw.index.get_level_values(1)
def get_prc_atExpiration(point):
    idx = stockindex == point.name[1]
    idx &= dateindex < point.expiration_date
    return prices_raw.loc[idx, 'prc'].sort_index(axis=0, level=1).iloc[-1]

downsampled_df['prc_atExpiration'] = downsampled_df.apply(get_prc_atExpiration, axis=1)

downsampled_df.comnam = downsampled_df.comnam.astype('str')
downsampled_df.ticker = downsampled_df.ticker.astype('str')

print('Sorting index')
data = downsampled_df.sort_index()


print('data.shape: {}'.format(data.shape))

'''
print('Splitting data into train, validation & test sets')
train, validate, test = np.split(merged.sample(frac=1, random_state=69777), [int(.6 * len(merged)), int(.8 * len(merged))])
print('{} - {} - {}'.format(train.shape[0], validate.shape[0], test.shape[0]))
'''

def generate_synthetic_data(option_type='call'):
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
        'vix'

        'prc_shifted_1',
        'option_price_shifted_1',
        'perfect_hedge_1'
    ]


    [
        'predicted_price',
        'predicted_hedge',
        'implied_delta'
    '''
    if option_type != 'call' and option_type != 'put':
        ValueError("option_type must be either 'call' or 'put'")

    # additional_columns = ['roe', 'roa', 'capital_ratio', 'pe_op_basic', 'pe_op_dil']
    additional_columns = optional_features



    means = {}
    stds = {}

    for column in additional_columns:
        means[column] = data[column].mean()
        stds[column] = data[column].std()

    impl_volatility = hist_impl_volatility = None
    complete_dummy_list = ['ff_ind_{}'.format(i) for i in range(49)]

    synth_list = []
    print('Generating synthetic contracts at maturity')
    S = 100
    for K in range(10, int(S * 1.5), 10):
        for i in range(0, 10):
            days = 0
            additional_column_values = [np.random.randn() * stds[column] + means[column] for column in additional_columns]

            if option_type == 'call':
                option_price = max(0, S - K)
                if S > K:
                    delta = 1
                elif S < K:
                    delta = 0
                else:
                    delta = 0.5
            else:
                option_price = max(0, K - S)
                if S > K:
                    delta = 0
                elif S < K:
                    delta = -1
                else:
                    delta = -0.5
            strike_price = K
            prc = S


            moneyness = S / K
            scaled_option_price = option_price / K

            dummy_values = [0] * len(complete_dummy_list)
            dummy_values[np.random.randint(49)] = 1

            new_row = [days, option_price, impl_volatility, hist_impl_volatility, delta, strike_price, prc, moneyness, scaled_option_price]
            new_row += dummy_values
            new_row += additional_column_values

            synth_list.append(new_row)

    print('Generating synthetic contracts at boundary condition S = 0 | S << K')
    K = 100
    S = 0
    for i in range(3):
        for days in range(2, 60):
            days = days / 365
            strike_price = K
            prc = S
            moneyness = 0
            additional_column_values = [np.random.randn() * stds[column] + means[column] for column in additional_columns]
            r = additional_column_values[additional_columns.index('r')]
            if option_type == 'call':
                # Sketchy: risk-free rate should be interpolated to match maturity
                option_price = 0
                delta = 0
            else:
                option_price = strike_price * np.exp(-r * days)
                delta = -1

            scaled_option_price = option_price / K

            dummy_values = [0] * len(complete_dummy_list)
            dummy_values[np.random.randint(49)] = 1
            new_row = [days, option_price, impl_volatility, hist_impl_volatility, delta, strike_price, prc, moneyness, scaled_option_price]
            new_row += dummy_values
            new_row += additional_column_values

            synth_list.append(new_row)

    print('Generating synthetic contracts at boundary condition S >> K')
    K = 100
    for S in range(int(K * 3.5), int(K * 6), 50):
        for days in range(2, 60):
            days = days / 365

            strike_price = K
            prc = S
            moneyness = S / K
            additional_column_values = [np.random.randn() * stds[column] + means[column] for column in additional_columns]
            r = additional_column_values[additional_columns.index('r')]

            if option_type == 'call':
                # Sketchy: risk-free rate should be interpolated to match maturity
                option_price = S - K * np.exp(-r * days)
                delta = 1
            else:
                option_price = 0
                delta = 0

            scaled_option_price = option_price / K

            dummy_values = [0] * len(complete_dummy_list)
            dummy_values[np.random.randint(49)] = 1
            new_row = [days, option_price, impl_volatility, hist_impl_volatility, delta, strike_price, prc, moneyness, scaled_option_price]
            new_row += dummy_values
            new_row += additional_column_values

            synth_list.append(new_row)

    synth_df = pd.DataFrame(synth_list, columns=['days', 'option_price', 'impl_volatility', 'hist_impl_volatility', 'delta', 'strike_price',
                                                 'prc', 'moneyness', 'scaled_option_price'] + complete_dummy_list + additional_columns)
    # Shuffling the order of the rows
    synth_df = synth_df.sample(frac=1)

    return synth_df


synth_df = generate_synthetic_data()

print('Storing result on disc')
with pd.HDFStore(paths['options_for_ann']) as store:
    # store['options_for_ann'] = merged
    # merged = store['options_for_ann']
    # store['train'] = train
    # store['validate'] = validate
    # store['test'] = test
    # store['single'] = single_stock
    store['data'] = data
    store['synthetic'] = synth_df
    store['availability_summary'] = availability_summary

print('Done')

sys.exit()