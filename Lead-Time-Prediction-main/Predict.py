from parameters import cache_location, meta_params, outlier_params, box_plot, forest, confidence
from cache_fct_to_disc import cache_to_disk
from utils import remove_label_outliers, visualize_outliers, dict_as_attr, stats as statistics
import macros as macros

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime as dt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import warnings
warnings.filterwarnings("ignore")


def get_params(params):
    return dict_as_attr(params)


filehandle = open("Config.txt","r")
code_to_execute = filehandle.read()
exec(code_to_execute)


meta_params = dict_as_attr(meta_params)
outlier_params = dict_as_attr(outlier_params)

print('Loading data..')
data = macros.open_data(cache_location, 'data.pcl')

" LOCFR, LOCTO, MATERIAL, MOT, DAYS, LEADTIME "
data_timeseries = data[[meta_params.date] + meta_params.features + [meta_params.label]]

data_history = data_timeseries[data_timeseries[meta_params.date] <= dt.datetime.strptime(meta_params.today, meta_params.date_format)]

@cache_to_disk
def transform(data, time_idx, date_col, date_format, label, granularity, min_group_size):
    print('Transforming time series..')
    date_series = pd.to_datetime((data[date_col].dt.strftime(date_format)), format=date_format)
    first_date = pd.to_datetime(min(date_series))

    data['YEAR'] = date_series.dt.strftime('%Y')
    data['MONTH'] = date_series.dt.strftime('%m')
    data['WEEKDAY'] = date_series.dt.weekday
    data['DAYS'] = (date_series - first_date).dt.days

    first_date_rounded_to_monday = first_date - pd.to_timedelta(first_date.weekday(), unit='d')
    date_rounded_to_monday = date_series - pd.to_timedelta(data['WEEKDAY'], unit='d')
    data['WEEKS'] = (date_rounded_to_monday - first_date_rounded_to_monday).dt.days / 7

    timeseries = data.sort_values(by=[time_idx])
    timeseries = timeseries.groupby(granularity + [time_idx]).mean().reset_index()[granularity + [time_idx, label]]
    counts = data.groupby(granularity + [time_idx]).size().reset_index(name='counts')
    timeseries = timeseries.merge(counts, how='left', on=granularity + [time_idx])
    timeseries = timeseries[timeseries['counts'] >= min_group_size]

    return timeseries


history = transform(data=data_timeseries, time_idx=meta_params.time_level, date_col=meta_params.date,
                       date_format=meta_params.date_format, label=meta_params.label,
                       granularity=meta_params.granularity, min_group_size=meta_params.min_group_size)

history = history[history[meta_params.time_level] >= max(history[meta_params.time_level]) - meta_params.history]
history[meta_params.time_level] = history[meta_params.time_level] - min(history[meta_params.time_level])
history.to_excel(Path.joinpath(cache_location, 'Timeseries.xlsx'), merge_cells=False, index=False)

""" Remove Outliers """
if meta_params.outliers:
    print('Processing outliers..')
    filter = None
    @cache_to_disk
    def process_label_outliers(data, label, cap=None, box_plot=None, forest=None, confidence=None, remove=True):
        data_wo_outliers, outliers, outlier_vis_data = remove_label_outliers(data, label, cap, box_plot, forest,
                                                                             confidence, remove)
        return data_wo_outliers, outliers, outlier_vis_data

    if outlier_params.outlier_removal_method == 'box_plot':
        data, outliers, outlier_vis_data = process_label_outliers(data=history,
                                                                  label=meta_params.label,
                                                                  cap=outlier_params.cap,
                                                                  box_plot=box_plot,
                                                                  forest=None,
                                                                  confidence=None,
                                                                  remove=outlier_params.remove_outliers)

    if outlier_params.outlier_removal_method == 'forest':
        data, outliers, outlier_vis_data = process_label_outliers(data=history,
                                                                  label=meta_params.label,
                                                                  cap=outlier_params.cap,
                                                                  box_plot=None,
                                                                  forest=forest,
                                                                  confidence=None,
                                                                  remove=outlier_params.remove_outliers)

    if outlier_params.outlier_removal_method == 'confidence':
        data, outliers, outlier_vis_data = process_label_outliers(data=history,
                                                                  label=meta_params.label,
                                                                  cap=outlier_params.cap,
                                                                  box_plot=None,
                                                                  forest=None,
                                                                  confidence=confidence,
                                                                  remove=outlier_params.remove_outliers)

    if outlier_params.visualize_outliers_flag:

        visualize_outliers(show_outliers=True,
                           demo=meta_params.demo,
                           vis_data=outlier_vis_data,
                           box_plot=box_plot,
                           forest=None,
                           filter=filter,
                           sort_by_count=outlier_params.vis_sort_by_count,
                           title='Lane Distribution Box Plot',
                           x_axis='Granularity',
                           y_axis='Lead Time [d]')

        visualize_outliers(show_outliers=True,
                           demo=meta_params.demo,
                           vis_data=outlier_vis_data,
                           box_plot=None,
                           forest=forest,
                           filter=filter,
                           sort_by_count=outlier_params.vis_sort_by_count,
                           title='Lane Distribution Density',
                           x_axis='Granularity',
                           y_axis='Lead Time [d]')

    outliers.to_excel(Path.joinpath(cache_location, 'Outliers.xlsx'), merge_cells=False, index=False)

if meta_params.analyze:
    print('Calculating statistics..')
    statistics(history, meta_params.label, key=meta_params.granularity, mode='short', demo=meta_params.demo,
               sort_by_count=outlier_params.vis_sort_by_count, location=cache_location, save=True,
               confidence=meta_params.confidence, file='Statistics.xlsx',
               chart=meta_params.chart, min_group_size=meta_params.min_group_size,
               value_label='Lead Time [d]')


def predict_ts(timeseries, time_idx, granularity, label, loc_fr=[], loc_to=None):
    print('Retrieving prediction..')
    if loc_to is not None:
        timeseries = timeseries[timeseries['DestinationSite'].isin(loc_to)]
    for l in loc_fr:
        timeseries_filtered = timeseries[timeseries['OriginSite'] == l]

        timeseries_filtered = timeseries_filtered.groupby(granularity).agg(list)  # .reset_index()
        timeseries_filtered = timeseries_filtered.head(5)

        for i in range(len(timeseries_filtered)):
            x = pd.to_numeric(timeseries_filtered[time_idx].values[i])
            y = pd.to_numeric(timeseries_filtered[label].values[i])
            plt.plot(x, y)
        plt.show()

        for i in range(len(timeseries_filtered)):
            series = pd.to_numeric(timeseries_filtered[label].values[i])
            x = pd.to_numeric(timeseries_filtered[time_idx].values[i])
            '''
            model = ARIMA(series, order=(5,1,0))
            model_fit = model.fit()
            print(model_fit.summary())
            '''

            X = series
            size = int(len(X) * 0.66)
            train, test = X[0:size], X[size:len(X)]
            history = [x for x in train]
            predictions = list()

            ''' RECURSIVE '''
            '''
            for t in range(len(test)):
                model = ARIMA(history, order=(5, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)

            rmse = math.sqrt(mean_squared_error(test, predictions))
            print('Test RMSE: %.3f' % rmse) 
            plt.plot(test)
            plt.plot(predictions, color='red')
            '''

            ''' STRAIGHT '''
            horizon = min(30, len(test))
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast(steps=horizon)
            yhat = output[0]
            predictions.append(yhat)
            test = test[:horizon]
            rmse = math.sqrt(mean_squared_error(test, predictions[0]))
            print('Test RMSE: %.3f' % rmse)
            x_past = x[:size]
            x_future = x[size:]
            x_future = x_future[:horizon]
            plt.plot(x_past, train)

            plt.plot(x_future, test)
            plt.plot(x_future, predictions[0], color='red')

            plt.show()


if meta_params.predict:
    predict_ts(history, meta_params.time_level, meta_params.granularity, meta_params.label, loc_fr=['1002'])