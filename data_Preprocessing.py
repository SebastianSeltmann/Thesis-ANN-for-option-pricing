import pandas as pd
import numpy as np
import quandl
import wrds as wrds
import datetime

from config import paths, quandl_key
from reused_code import fetch_and_store_sp500, store_options
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


quandl.ApiConfig.api_key = quandl_key


# Price Data
print('Loading Stock Prices')
store = pd.HDFStore(paths['h5 constituents & prices'])
prices_raw = store['Prices_raw']
prices = store['Prices']
store.close()
print(prices_raw.count())

# Rolling Returns & Vola Calculation
'''# problems with set mismatch
melted_returns.set_index(['date','permno'],inplace=True)
prices_raw.set_index(['date','permno'],inplace=True)

mismatch = melted_returns[~melted_returns.index.isin(prices_raw.index)]
melted_returns[melted_returns.index.isin(mismatch)].dropna()

t1 = melted_returns[['date','permno']]
t2 = prices_raw[['date','permno']]
t1['date'] = t1['date'].astype('datetime64[ns]')
t1['permno'] = t1['permno'].astype(np.int64)
t2['date'] = t2['date'].astype('datetime64[ns]')
t2['permno'] = t2['permno'].astype(np.int64)

common = t1.merge(t2,on=['date','permno'])
t1[(~t1.date.isin(common.date))&(~t1.permno.isin(common.permno))]
melted_returns.count()
t1.set_index(['date','permno'], inplace=True)
t1.index
'''

returns = prices.pct_change()

def reshape_into_series(df):
    df.index = df.index.astype('datetime64[ns]')
    df.columns = df.columns.astype(np.int64)
    ser = df.unstack().dropna()
    return ser.swaplevel()

ser_returns = reshape_into_series(returns.rolling(110).mean() * 252)
ser_v110 = reshape_into_series(returns.rolling(110).std() * np.sqrt(252))
ser_v60  = reshape_into_series(returns.rolling(60).std() * np.sqrt(252))
ser_v20  = reshape_into_series(returns.rolling(20).std() * np.sqrt(252))
ser_v5   = reshape_into_series(returns.rolling(5).std() * np.sqrt(252))


def compute_optionsdata():
    store_options(option_type='call')

# Options Data
print('Loading Options')
def get_optionsdata_for_year(year):
    store = pd.HDFStore(paths['all_options_h5'])
    optionsdata_for_year = store['options' + str(year)]
    store.close()
    return optionsdata_for_year

start_year = 1996
end_year = 2016 # up to 2016 ?

# start_year = 2010
# end_year = 2011
df = pd.DataFrame()
for year in range(start_year, end_year):  # range(1996, 2016)
    options_data_year = get_optionsdata_for_year(year)
    options_data_year.rename(index=str, columns={"id": "permno"}, inplace = True)
    options_data_year.set_index(['date','permno'],inplace=True)
    df = df.append(options_data_year)


print(df.count())

# risk-free rate
print('Loading risk-free rate')
def download_treasury_data():
    treasury = quandl.get("FRED/DGS3MO") # 3-Month Treasury Constant Maturity Rate
    store = pd.HDFStore(paths['treasury'])
    store['treasury'] = treasury
    store.close()
store = pd.HDFStore(paths['treasury'])
treasury = store['treasury']
store.close()

def download_vix_data():
    vix = quandl.get("CHRIS/CBOE_VX1") # S&P 500 Volatility Index VIX Futures
    store = pd.HDFStore(paths['vix'])
    store['vix'] = vix
    store.close()
store = pd.HDFStore(paths['vix'])
vix = store['vix']
store.close()

def download_dividends_data():
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


def download_fundamentals_data():
    print('Downloading fundamentals data')
    db = wrds.Connection()
    permnos = prices_raw.permno.drop_duplicates()

    query = ("select * "
             "from wrdsapps.firm_ratio "
             "where public_date > '" + str(start_year) + "0101' "
             "and public_date < '" + str(end_year) + "0101' "
             "and permno in (" + ','.join(permnos.astype(str)) + ")")
    print('Starting Query')
    firm_ratios = db.raw_sql(query)
    print('Finished Query')

    store = pd.HDFStore(paths['ratios'])
    store['ratios'] = firm_ratios
    store.close()

store = pd.HDFStore(paths['ratios'])
ratios = store['ratios']
store.close()

def download_names_data():
    print('Downloading names data')
    db = wrds.Connection()
    permnos = prices_raw.permno.drop_duplicates()
    start_year = 1996
    end_year = 2016
    query = ("select permno, comnam, ticker, namedt, nameenddt "
             "from crspa.stocknames "
             "where nameenddt > '" + str(start_year) + "0101' "
             "and namedt < '" + str(end_year) + "0101' "
             "and permno in (" + ','.join(permnos.astype(str)) + ")")
    print('Starting Query')
    names = db.raw_sql(query)
    print('Finished Query')

    store = pd.HDFStore(paths['names'])
    store['names'] = names
    store.close()

store = pd.HDFStore(paths['names'])
names = store['names']
store.close()


def determine_available_data():
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

store = pd.HDFStore(paths['dividends'])
dividends = store['dividends']
store.close()

# Merged Data
print('Merging with prices data')
prices_raw['permno'] = prices_raw['permno'].astype(np.int64)
prices_raw['date'] = prices_raw['date'].astype('datetime64[ns]')
prices_raw.set_index(['date','permno'],inplace=True)

prices_raw['prc_shifted_1'] = prices_raw.groupby(level=1)['prc'].shift(-1)


merged = pd.merge(df[['days', 'option_price', 'impl_volatility', 'delta', 'strike_price']],
         prices_raw[['prc', 'prc_shifted_1']], how='inner', left_index=True, right_index=True)

# del prices_raw
# del df

'''
merged = pd.merge(df[['id', 'date', 'days', 'option_price', 'impl_volatility', 'delta', 'strike_price']],
         prices_raw[['date', 'permno', 'prc']], how='inner', left_on=['date', 'id'], right_on=['date', 'permno'])
'''

print('Merging with returns data')
merged['returns'] = ser_returns
merged['v110'] = ser_v110
merged['v60'] = ser_v60
merged['v20'] = ser_v20
merged['v5'] = ser_v5

print('Merging with treasury and vix data')
#mergedx = pd.merge(mini_merged, treasury, how='inner', left_on='date', right_index=True)
merged = pd.merge(merged.reset_index(), treasury.reset_index(), left_on='date', right_on='Date', how='inner')
merged = pd.merge(merged, vix.loc[:,'Close'].reset_index(), left_on='date', right_on='Trade Date', how='inner')
merged.drop(['Date', 'Trade Date'], axis=1, inplace=True)
merged.rename(index=str, columns={"Value": "r", "Close": "vix"}, inplace=True)

print('Merging with fundamentals data')
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
fundamental_columns_to_include = ratios.columns
ratios_to_merge = ratios.loc[:,fundamental_columns_to_include]

dummies = pd.get_dummies(ratios_to_merge.ffi49.fillna(0).astype(int), prefix='ff_ind')
complete_dummy_list = ['ff_ind_{}'.format(i) for i in range(49)]
for dummy in complete_dummy_list:
    if dummy not in dummies.columns:
        dummies[dummy] = 0

ratios_to_merge = pd.concat([ratios_to_merge, dummies], axis=1)

def get_last_day_of_month(date):
    import calendar
    last_day = calendar.monthrange(date.year, date.month)[1]
    return datetime.date(date.year, date.month, last_day)

merged['month_end_date'] = merged.loc[:,'date'].apply(get_last_day_of_month)
merged = pd.merge(merged, ratios_to_merge, left_on=['permno','month_end_date'], right_on=['permno', 'public_date'],
                  how='inner')
merged.drop(columns=['month_end_date', 'public_date'], inplace=True)

print('Merging with names data')
idx = names.groupby(['permno'])['nameenddt'].transform(max) == names['nameenddt']
last_names = names.loc[idx, ['permno', 'comnam', 'ticker']]
merged = pd.merge(merged, last_names, left_on=['permno'], right_on=['permno'], how='inner')


print('Computing Expiration Date & shifting option_price')
merged.reset_index(inplace=True)
merged['timedelta'] = merged.loc[:, 'days'].apply(datetime.timedelta)
merged['expiration_date'] = pd.to_datetime(merged['date']) + merged.loc[:, 'timedelta']
merged.drop(columns=['timedelta'], inplace=True)

merged.set_index(['date', 'permno', 'strike_price', 'expiration_date'], inplace=True)
merged['option_price_shifted_1'] = merged.groupby(level=[1, 2, 3])['option_price'].shift(-1)
merged.reset_index(inplace=True)
merged.set_index(['date', 'permno'], inplace=True)

print('Dropping NaN values: {:.2f}%'.format(merged.isna().any(axis=1).mean()*100))
merged.dropna(how='any', inplace=True)

print('Computing moneyness')
merged['moneyness'] = merged.prc / merged.strike_price
merged['scaled_option_price'] = merged.option_price / merged.strike_price
merged['scaled_option_price_shifted_1'] = merged.option_price_shifted_1 / merged.strike_price

print('Computing perfect hedge (with hindsight)')
merged.loc[:, 'perfect_hedge_1'] = -( merged.loc[:, 'prc'] - merged.loc[:, 'prc_shifted_1']
                                      ) / (
                                    merged.loc[:, 'option_price'] - merged.loc[:, 'option_price_shifted_1'])

merged = merged.replace([np.inf, -np.inf], 0) # Sketchy

merged.loc[:, 'P_value_change_1'] = (merged.loc[:, 'prc_shifted_1'] - merged.loc[:, 'prc']
                                     )+(
                                        merged.loc[:, 'perfect_hedge_1'] *
                                        (merged.loc[:, 'option_price_shifted_1'] - merged.loc[:, 'option_price'])
                                    )


print(merged.shape)

print('Splitting data into train, validation & test sets')
train, validate, test = np.split(merged.sample(frac=1, random_state=69777), [int(.6 * len(merged)), int(.8 * len(merged))])

print('{} - {} - {}'.format(train.shape[0], validate.shape[0], test.shape[0]))


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

    additional_columns = ['roe', 'roa', 'capital_ratio', 'pe_op_basic', 'pe_op_dil']



    means = {
        'returns': train.returns.mean(),
        'v110': train.v110.mean(),
        'v60': train.v60.mean(),
        'v20': train.v20.mean(),
        'v5': train.v5.mean(),
        'r': train.r.mean(),
        'vix': train.vix.mean(),
    }

    stds = {
        'returns': train.returns.std(),
        'v110': train.v110.std(),
        'v60': train.v60.std(),
        'v20': train.v20.std(),
        'v5': train.v5.std(),
        'r': train.r.std(),
        'vix': train.vix.std(),
    }
    for column in additional_columns:
        means[column] = train[column].mean()
        stds[column] = train[column].std()

    complete_dummy_list = ['ff_ind_{}'.format(i) for i in range(49)]

    synth_list = []
    print('Generating synthetic contracts at maturity')
    S = 100
    for K in range(10, int(S * 1.5), 10):
        for i in range(0, 10):
            days = 0
            impl_volatility = None
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
            returns = np.random.randn() * stds['returns'] + means['returns']
            v110 = np.random.randn() * stds['v110'] + means['v110']
            v60 = np.random.randn() * stds['v60'] + means['v60']
            v20 = np.random.randn() * stds['v20'] + means['v20']
            v5 = np.random.randn() * stds['v5'] + means['v5']
            r = np.random.randn() * stds['r'] + means['r']
            vix = np.random.randn() * stds['vix'] + means['vix']

            additional_column_values = [np.random.randn() * stds[column] + means[column] for column in additional_columns]

            moneyness = S / K
            scaled_option_price = option_price / K

            dummy_values = [0] * len(complete_dummy_list)
            dummy_values[np.random.randint(49)] = 1

            new_row = [days, option_price, impl_volatility, delta, strike_price, prc, returns, v110, v60, v20,
                               v5, r, moneyness, scaled_option_price, vix]
            new_row += dummy_values
            new_row += additional_column_values

            synth_list.append(new_row)

    print('Generating synthetic contracts at boundary condition S = 0 | S << K')
    K = 100
    S = 0
    for i in range(3):
        for days in range(2, 60):
            days = days / 365
            impl_volatility = None
            strike_price = K
            prc = S
            returns = np.random.randn() * stds['returns'] + means['returns']
            v110 = np.random.randn() * stds['v110'] + means['v110']
            v60 = np.random.randn() * stds['v60'] + means['v60']
            v20 = np.random.randn() * stds['v20'] + means['v20']
            v5 = np.random.randn() * stds['v5'] + means['v5']
            r = np.random.randn() * stds['r'] + means['r']
            vix = np.random.randn() * stds['vix'] + means['vix']
            moneyness = 0
            additional_column_values = [np.random.randn() * stds[column] + means[column] for column in additional_columns]

            if option_type == 'call':
                # Sketchy: risk-free rate should be interpolated to match maturity
                option_price = 0 * np.exp(-r * days)
                delta = 0
            else:
                option_price = strike_price * np.exp(-r * days)
                delta = -1

            scaled_option_price = option_price / K

            dummy_values = [0] * len(complete_dummy_list)
            dummy_values[np.random.randint(49)] = 1
            new_row = [days, option_price, impl_volatility, delta, strike_price, prc, returns, v110, v60, v20,
                               v5, r, moneyness, scaled_option_price, vix]
            new_row += dummy_values
            new_row += additional_column_values

            synth_list.append(new_row)

    print('Generating synthetic contracts at boundary condition S >> K')
    K = 100
    for S in range(int(K * 2.5), int(K * 4), 50):
        print('{} - {}'.format(S, S/K))
        for days in range(2, 60):
            days = days / 365
            impl_volatility = None

            strike_price = K
            prc = S
            returns = np.random.randn() * stds['returns'] + means['returns']
            v110 = np.random.randn() * stds['v110'] + means['v110']
            v60 = np.random.randn() * stds['v60'] + means['v60']
            v20 = np.random.randn() * stds['v20'] + means['v20']
            v5 = np.random.randn() * stds['v5'] + means['v5']
            r = np.random.randn() * stds['r'] + means['r']
            vix = np.random.randn() * stds['vix'] + means['vix']
            moneyness = S / K
            additional_column_values = [np.random.randn() * stds[column] + means[column] for column in additional_columns]

            if option_type == 'call':
                # Sketchy: risk-free rate should be interpolated to match maturity
                option_price = S * np.exp(-r * days)
                delta = 1
            else:
                option_price = 0
                delta = 0

            scaled_option_price = option_price / K

            dummy_values = [0] * len(complete_dummy_list)
            dummy_values[np.random.randint(49)] = 1
            new_row = [days, option_price, impl_volatility, delta, strike_price, prc, returns, v110, v60, v20,
                               v5, r, moneyness, scaled_option_price, vix]
            new_row += dummy_values
            new_row += additional_column_values

            synth_list.append(new_row)

    synth_df = pd.DataFrame(synth_list, columns=['days', 'option_price', 'impl_volatility', 'delta', 'strike_price',
                                                 'prc', 'returns', 'v110', 'v60', 'v20', 'v5', 'r', 'moneyness',
                                                 'scaled_option_price', 'vix'] + complete_dummy_list + additional_columns)
    # Shuffling the order of the rows
    synth_df = synth_df.sample(frac=1)

    return synth_df


synth_df = generate_synthetic_data()


print('Singling out some stock')
some_stocks = merged.index.levels[1][0:10]
print(some_stocks)

# some_stocks = ['10107', '81774', '14542']
some_stock = some_stocks[3]
single_stock = merged.loc[(slice(None), some_stock),:]

# single_train, single_validate, single_test = np.split(single_stock.sample(frac=1, random_state=69777), [int(.8 * len(single_stock)), int(.9 * len(single_stock))])


print('Storing result on disc')
store = pd.HDFStore(paths['options_for_ann'])
# store['options_for_ann'] = merged
# merged = store['options_for_ann']
store['train'] = train
store['validate'] = validate
store['test'] = test
store['single'] = single_stock
store['synthetic'] = synth_df
store.close()

print('Done')

def determine_perfect_hedges_with_hindsight(merged=None):
    mini_merged = merged[0:1000]
    mini_merged = merged.loc[(slice('2010-01-05 00:00:00'),['10078'],[11.0]),:]



    mini_merged.drop(columns=['P_value_change_1', 'perfect_hedge_1'], inplace=True)

    merged.reset_index(inplace=True)
    merged.set_index(['date', 'permno', 'strike_price'], inplace=True)


    filtered = mini_merged
    cols = ['prc', 'prc_shifted_1', 'option_price', 'option_price_shifted_1', 'perfect_hedge_1', 'P_value_change_1']

    print(mini_merged.loc[('2010-01-04 00:00:00', '10078', 11.0),cols].values)
    print(mini_merged.loc[('2010-01-05 00:00:00', '10078', 11.0),cols].values)
    print(mini_merged.loc[('2010-01-06 00:00:00', '10078', 11.0),cols].values)



    print(merged.loc[('2010-01-04 00:00:00', '10078', 11.0):('2010-01-06 00:00:00', '10078', 11.0),cols])
    print(merged.loc['2010-01-04 00:00:00', '10078', 11.0][cols])
    print(merged.loc['2010-01-05 00:00:00', '10078', 11.0][cols])
    print(merged.loc['2010-01-06 00:00:00', '10078', 11.0][cols])




    merged = merged[['days', 'option_price', 's1', 'impl_volatility', 'delta', 'prc',
       'prc_shifted_1', 'returns', 'v110', 'v60', 'v20', 'v5', 'r', 'vix',
       'moneyness', 'scaled_option_price']]


def experimentation(MinMaxScaler=None):
    mini_merged = merged[0:1000]
    temp = merged.loc[(slice(None),10078),:]

    print('rescaling data')
    print('actually, not really')
    def scale(train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    mini = test[0:1000]
    train.columns
    train.days.max()