import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import os

from config import paths

def actual_vs_fitted_plot(model, prediction_input_data, prediction_target, segment_plot, X_val, Y_prediction,
                          sample_size, offset):
    alpha = 0.3
    score = model.evaluate(prediction_input_data, prediction_target, verbose=2)
    print(score)

    # global Y_prediction

    if not segment_plot:
        plt.plot(prediction_target.scaled_option_price, Y_prediction, "+",
                 label='{:1.0f}'.format(offset / sample_size + 1))
    else:
        in_the_money = X_val['moneyness'] < 0.9
        out_the_money = X_val['moneyness'] > 1.1
        at_the_money = ~in_the_money & ~out_the_money
        plt.plot(prediction_target.scaled_option_price[in_the_money], Y_prediction[in_the_money], "o",
                 label='in the money', alpha=alpha)
        plt.plot(prediction_target.scaled_option_price[at_the_money], Y_prediction[at_the_money], "o",
                 label='at the money', alpha=alpha)
        plt.plot(prediction_target.scaled_option_price[out_the_money], Y_prediction[out_the_money], "o",
                 label='out the money', alpha=alpha)

    plt.show()

def get_data_description():
    from data import data
    from config import full_feature_combination_list

    cols = ['option_price'] + full_feature_combination_list[-1]
    relevant_data = data.loc[:, cols]
    relevant_data.loc[:, 'days'] = np.int_(relevant_data.loc[:, 'days'])

    desc_df = pd.DataFrame(index=relevant_data.columns)
    desc_df['mean'] = relevant_data.mean()
    desc_df['min'] = relevant_data.min()
    desc_df['25%'] = relevant_data.quantile(0.25)
    desc_df['50%'] = relevant_data.quantile(0.50)
    desc_df['75%'] = relevant_data.quantile(0.75)
    desc_df['max'] = relevant_data.max()
    desc_df['variance'] = relevant_data.var()
    desc_df['skewness'] = relevant_data.skew()
    desc_df['kurtosis'] = relevant_data.kurtosis()

    '''
    tex = desc_df.round(2).to_latex()
    with open('description-tex.txt', 'w') as text_file:
        text_file.write(tex)
    '''
    print(desc_df)


def heatmapplot_correlations():
    from matplotlib import pyplot as plt
    import numpy as np
    from data import data
    from config import full_feature_combination_list
    columns_to_be_included_in_covar_calc = ['option_price'] + full_feature_combination_list[-1]
    relevant_data = data.loc[:,columns_to_be_included_in_covar_calc]
    relevant_data.loc[:, 'days'] = np.int_(relevant_data.loc[:,'days'])

    covars = relevant_data.cov()
    corrs = relevant_data.corr()

    def drop_leading_zero(val):
        if val.startswith("0."):
            return val[1:]
        if val.startswith("-0."):
            return "-" + val[2:]
        return val

    fig, ax = plt.subplots()
    cax = ax.imshow(corrs, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(corrs.columns)))
    ax.set_yticks(np.arange(len(corrs.index)))
    ax.set_xticklabels(list(corrs.columns), rotation=90)
    ax.set_yticklabels(list(corrs.index))
    for i in range(len(corrs.columns)):
        for j in range(len(corrs.index)):
            # num = '{:d}'.format(int(corrs.iloc[i, j]*100))
            num = drop_leading_zero('{:.2f}'.format(corrs.iloc[i,j]))
            text = ax.text(j, i, num,
                           ha="center", va="center", color="k")
    # ax.set_title('Correlation Matrix of included features')
    plt.savefig('plots/correlation-matrix.png', bbox_inches="tight")
    fig.show()

def scatterplot_PAD(model, datasets, id):
    from actions import get_gradients
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    axes = axes.flatten()
    for points in datasets:
        gradients = get_gradients(model, points)

        for i, var in enumerate(points.columns):
            axes[i].set_title(var)
            x = np.array(points.iloc[:, i])
            y = gradients[:, i]
            axes[i].scatter(x, y, alpha=0.1, marker='o', s=0.3)  # alpha=0.01
    for unneeded in range(i + 1, len(axes)):
        fig.delaxes(axes[unneeded])
    plt.tight_layout()

    if id > 0:
        filename = 'plots/Partial_derivatives-scatter_{}.png'.format(id)
    else:
        filename = 'plots/Partial_derivatives-scatter.png'
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

def get_SSD(model, inputs):
    from actions import get_gradients
    gradients_of_individual_inputs = get_gradients(model, inputs)
    SSD = np.square(gradients_of_individual_inputs).mean(axis=0)
    return SSD


def boxplot_SSD_distribution(SSD_distribution, features, set, model_name):
    _features = features.copy()

    SSDD_df = pd.DataFrame(SSD_distribution, columns=_features)

    contributions_df = SSDD_df.divide(SSDD_df.sum(axis=1), axis=0)

    c_df_without_moneyness = contributions_df.drop('moneyness', axis=1)
    _features.remove('moneyness')

    plt.figure()
    plt.title(set + '\n' + model_name)
    plt.boxplot(c_df_without_moneyness.transpose(), labels=_features, vert=False)
    plt.savefig('plots/SSD-dist-{}-{}.png'.format(set, model_name), bbox_inches="tight")
    plt.show()


def moving_average(a, n=3):
    '''
    moving_average(abs_error)
    moving_average(abs_error,n=100)
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_error_over_epochs():
    global loss, abs_error
    plt.plot(abs_error, label='abs_error')
    plt.plot(moving_average(abs_error,n=100), label='abs_error')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(loss, color='blue', lw=2)
    ax.set_yscale('log')
    ax.set_title('ANN loss over epochs on logplot')
    fig.show()


def plot_surface_of_ANN(model):
    h = 50
    w = 50
    ttm = np.array(range(0, h)) / (h/2)
    moneyness = np.array(range(0, w)) / (h/2)
    X = pd.DataFrame(columns=moneyness, index=ttm)
    for t in range(h):
        for m in range(w):
            X.iloc[t,m] = model.predict(np.array([[ttm[t], moneyness[m]]]))[0][0]

    '''
    d = {'moneyness': moneyness, 'ttm': ttm}
    X = pd.DataFrame(data=d)
    predicted_price = model.predict(X)
    '''
    data = [dict(
            type='surface',
            # x = X.columns.values,
            # y = X.index.values,
            z=X.as_matrix()
        )]
    layout = go.Layout(
        title='Pricing Surface predicted by ANN',
        autosize=False,
        width=800,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50
        ),

        scene=dict(
            xaxis=dict(title='Moneyness'),
            yaxis=dict(title='Time to Maturity'),
            zaxis=dict(title='Pricing')
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig)


def vol_surface_plot(input_data, setNames=None, variable='impl_volatility'):
    data = []

    if len(input_data) > 1:
        marker = dict(
            size=3,
            symbol='3',
            opacity=0.8
        )
    else:
        marker = dict(
            size=3,
            color=input_data[0][variable].values,
            colorscale='Viridis',
            symbol='3',
            opacity=0.8
        )

    for i, day in enumerate(input_data):
        x = day.days.values
        y = day.moneyness.values
        z = day[variable].values

        if setNames:
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=marker,
                name=setNames[i]
            )
        else:
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=marker
            )
        data.append(trace)

    layout = go.Layout(
        title=variable,
        autosize=False,
        width=900,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),

        scene=dict(
            xaxis=dict(title='Time to Maturity'),
            yaxis=dict(title='Moneyness'),
            zaxis=dict(title=variable)
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename=os.path.join('result-plots',  variable + '-'+setNames[0] + '.html'))


def get_and_plot(setNames, variable='error'):
    data = []
    store = pd.HDFStore(paths['neural_net_output'])
    for setName in setNames:
        data.append(store[setName])
    vol_surface_plot(input_data=data, setNames=setNames, variable=variable)
    store.close()

