from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import variation as cv
from functools import reduce
import pathlib
from functools import partial
from time import sleep
from tqdm import tqdm


class dict_as_attr(object):
    def __init__(self, d):
        self.__dict__ = d


def wrap_nan(val):
    if val == np.nan:
        return 0
    else:
        return val


def isin_filter_sorted(df, col, list):
    "same as dataframe isin filter, but the filtered df is sorted by the list"
    df_filtered = reduce(pd.DataFrame.append, map(lambda i: df[df[col] == i], list))
    return df_filtered


def stats(data, column, key=None, mode='full', demo=False, sort_by_count=True, location=None, file=None, save=False, confidence=0.95,
          chart=False, min_group_size=10, value_label='Value'):
    data[column] = data[column].astype(float)
    stats = pd.DataFrame()

    if key is not None:
        #l = data[key].apply(str).values
        #l = data[key].iloc[0, :].apply(str).values
        #s = "/".join(l)
        #stats['Group'] = s

        # grouped = data[column, key].groupby(key)
        #cv = np.std() / np.mean()

        quantile_25 = partial(np.quantile, q=.25)
        quantile_75 = partial(np.quantile, q=.75)
        confidence_from = partial(np.quantile, q=1 - confidence)
        confidence_to = partial(np.quantile, q=confidence)

        data.insert(loc=0, column="Group", value=data[key].agg(' / '.join, axis=1))

        group = dict.fromkeys(key, 'first')
        stat_col_dict = {column: [np.ma.count, np.mean, np.std, cv, np.median,
                  quantile_25, quantile_75, confidence_from, confidence_to]}
        stats = data.groupby('Group').agg({**group, **stat_col_dict})
        stats.columns = key + ['Count', 'Mean', 'Std', 'CV', 'Median', '25 Quantile', '75 Quantile', 'Confidence Lower Bound',
                         'Confidence Upper Bound']

        stats = stats[stats['Count'] >= min_group_size]
        if sort_by_count:
            stats = stats.sort_values(by='Count', ascending=False)
        else:
            stats = stats.sort_values(by='Group', ascending=True)

        if chart:
            sns.set_theme(style="whitegrid")
            #stats["Group"] = stats.index
            stats.insert(loc=0, column="Group", value=stats.index)
            #counts = vis_data.groupby(["Group"]).size().reset_index(name='count')
            if sort_by_count:
                stats = stats.sort_values(by='Count', ascending=False).reset_index(drop=True)
            uniques = stats["Group"]
            #uniques = stats.index
            chunks = np.array_split(uniques, np.ceil(len(uniques) / chart))
            n = 1
            for c in chunks:
                chunk_data = isin_filter_sorted(stats, 'Group', c)

                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
                #ax2 = ax1.twinx()
                #ax2.grid(False)
                #chunk_data = chunk_data.replace({'outliers': {1: 'in', -1: 'out'}})

                #g = sns.boxplot(x="Group", y="Value", data=chunk_data, ax=ax1)
                sns.set_theme()
                palette = {"Mean": 'cornflowerblue', "CV": 'red'}
                #h1 = sns.histplot(x="Group", hue="outliers", data=chunk_data, multiple='stack',
                #                  palette=palette, alpha=0.4, ax=ax2)

                df_mean = chunk_data[['Group', 'Mean']]
                df_mean = df_mean.rename(columns={"Mean": value_label})
                df_mean['characteristic'] = 'Mean'

                df_cv = chunk_data[['Group', 'CV']]
                df_cv = df_cv.rename(columns={"CV": value_label})
                df_cv['characteristic'] = 'CV'

                frames = [df_mean, df_cv]
                chart_data = pd.concat(frames)

                '''
                x = chunk_data['Mean']
                y = chunk_data['CV']
                z = chunk_data['Group']
                '''
                '''
                df = pd.concat(axis=0, ignore_index=True, objs=[
                    pd.DataFrame.from_dict({'value': stats['Mean'], 'name': 'Mean'}),
                    pd.DataFrame.from_dict({'value': stats['CV'], 'name': 'CV'}),
                ])

                series = pd.concat(axis=0, ignore_index=True, objs=[
                    pd.DataFrame.from_dict({'value': stats['Mean'], 'name': 'Mean'}),
                    pd.DataFrame.from_dict({'value': stats['CV'], 'name': 'CV'}),
                ])
                
                df['Group'] = stats['Group']
                '''


                h1 = sns.barplot(x='Group', y=value_label, hue='characteristic', data=chart_data,
                                 palette=palette, alpha=0.4, ax=ax1)
                #plt.hist([chunk_data['Mean'], chunk_data['CV']], color=['red','cornflowerblue'], alpha=0.4)
                plt.tight_layout()
                plt.title('Chart ' + str(n))
                ax1.set_xlabel('Granularity')
                plt.legend(loc='upper right')
                #ax1.set_ylabel('y')
                plt.show()
                n += 1
                if demo:
                    if n == 3:
                        break

            stats = stats.drop('Group', axis=1)

    else:
        # no key provided
        qty = data[[column]]
        stats.loc['', 'mean'] = qty[column].mean()
        stats.loc['', 'std'] = qty[column].std()
        stats.loc['', 'CV'] = (qty[column].std() / qty[column].mean())

        if mode == 'full':
            stats.loc['', 'count'] = len(qty)
            stats.loc['', 'min'] = qty[column].min()
            stats.loc['', 'max'] = qty[column].max()
            stats.loc['', 'median'] = qty[column].median()
            stats.loc['', '25_quantile'] = qty[column].quantile(0.25)
            stats.loc['', '50_quantile'] = qty[column].quantile(0.5)
            stats.loc['', '75_quantile'] = qty[column].quantile(0.75)
            stats.loc['', '90_quantile'] = qty[column].quantile(0.90)
            stats.loc['', '95_quantile'] = qty[column].quantile(0.95)
            stats.loc['', '99_quantile'] = qty[column].quantile(0.99)

    #print(stats)
    if save:
        stats.columns = key + ['Count [nr]', 'Mean [d]', 'Std [d]', 'CV [unitless]', 'Median [d]', '25 Quantile [d]',
                               '75 Quantile [d]', 'Confidence Lower Bound [d]',
                               'Confidence Upper Bound [d]']
        if location:
            stats.to_excel(pathlib.Path.joinpath(location, file), merge_cells=False, index=False)
        else:
            stats.to_excel(file, merge_cells=False, index=False)


def remove_label_outliers(data, label, cap=None, box_plot=None, forest=None, confidence=None, remove=True, statistics=False):
    """
    key = ['LOCID', 'LOCIDTO']
    forest = {
        "group": key,
        "min_group_sample": 30,  # group size to consider
        "contamination": 0.01,  # default: 'auto'
        "max_group_outliers": None  # outlier size limit per group
    }
    """

    print('Data volume before removing outliers:')
    print(len(data.index))

    if cap:
        outliers_cap = data[data[label] > cap]
        outliers_cap['outliers'] = -1
        if statistics:
            print('Outlier statistics cap:')
            stats(data=outliers_cap, column=label, file='Statistics_Cap.xlsx')
        if remove:
            data = data[data[label] <= cap]
            data[label] = data[label].astype(float)

    vis_data = pd.DataFrame()

    if box_plot:
        if box_plot['group']:
            data = data.sort_values(by=box_plot['group'], ascending=True).reset_index(drop=True)
            data_processed = pd.DataFrame()
            box_plot_vis_data = pd.DataFrame()
            groups = data[box_plot['group']].drop_duplicates()
            for index, row in tqdm(groups.iterrows()):
                row = pd.DataFrame(row).transpose()
                filtered = data.merge(row, how='inner', on=box_plot['group'])
                filtered[label] = filtered[label].astype(float)

                box_plot_vis_group = pd.DataFrame()
                box_plot_vis_group['Value'] = filtered[label]
                l = row[box_plot['group']].iloc[0, :].apply(str).values
                s = "/".join(l)
                box_plot_vis_group['Group'] = s

                Q1 = filtered[label].quantile(0.25)
                Q3 = filtered[label].quantile(0.75)
                IQR = Q3 - Q1  # IQR is interquartile range.
                filter = (filtered[label] >= Q1 - 1.5 * IQR) & (filtered[label] <= Q3 + 1.5 * IQR)
                filtered['outliers'] = 1
                filtered.loc[~filter, 'outliers'] = -1

                if box_plot['min_group_sample']:
                    if wrap_nan(len(filtered)) < box_plot['min_group_sample']:
                        filtered['outliers'] = 1  # no outlier detection for small groups

                if box_plot['remove_small_groups']:
                    if wrap_nan(len(filtered)) <= box_plot['remove_small_groups']:
                        filtered['outliers'] = -1

                box_plot_vis_group['outliers'] = filtered['outliers']
                data_processed = pd.concat([data_processed, filtered], ignore_index=True)
                box_plot_vis_data = pd.concat([box_plot_vis_data, box_plot_vis_group], ignore_index=True)
                vis_data = box_plot_vis_data
            data = data_processed

        else:
            Q1 = data[label].quantile(0.25)
            Q3 = data[label].quantile(0.75)
            IQR = Q3 - Q1  # IQR is interquartile range.

            filter = (data[label] >= Q1 - 1.5 * IQR) & (data[label] <= Q3 + 1.5 * IQR)
            data['outliers'] = 1
            data.loc[~filter, 'outliers'] = -1

        data_outliers = data[data['outliers'] == -1]
        if remove:
            data = data[data['outliers'] == 1]
        data = data.drop(columns=['outliers'])
        if statistics:
            print('Outlier statistics box_plot:')
            stats(data=data_outliers, column=label, file='Statistics_Box_Plot.xlsx')

    if forest:
        Ifo = IsolationForest(random_state=0, contamination=forest['contamination'])
        if forest['group']:
            data = data.sort_values(by=forest['group'], ascending=True).reset_index(drop=True)
            data_processed = pd.DataFrame()
            forest_vis_data = pd.DataFrame()
            groups = data[forest['group']].drop_duplicates()
            for index, row in tqdm(groups.iterrows()):
                row = pd.DataFrame(row).transpose()
                filtered = data.merge(row, how='inner', on=forest['group'])

                forest_vis_group = pd.DataFrame()
                forest_vis_group['Value'] = filtered[label]
                l = row[forest['group']].iloc[0, :].apply(str).values
                s = "/".join(l)
                forest_vis_group['Group'] = s

                outliers = Ifo.fit_predict(filtered[label].to_numpy().reshape(-1, 1))
                if forest['min_group_sample']:
                    if wrap_nan(len(filtered)) >= forest['min_group_sample']:
                        if forest['max_group_outliers']:
                            len_outliers = wrap_nan(len(outliers))
                            if len_outliers > forest['max_group_outliers']:  # more outliers for group than expected
                                saturated_contamination = forest['max_group_outliers'] / len_outliers
                                Ifo_calibrated = IsolationForest(random_state=0, contamination=saturated_contamination)
                                outliers = Ifo_calibrated.fit_predict(filtered[label].to_numpy().reshape(-1, 1))
                        filtered['outliers'] = outliers
                    else:
                        filtered['outliers'] = 1  # no outlier detection for small groups
                else:
                    filtered['outliers'] = outliers

                if forest['remove_small_groups']:
                    if wrap_nan(len(filtered)) <= forest['remove_small_groups']:
                        filtered['outliers'] = -1

                forest_vis_group['outliers'] = filtered['outliers']
                data_processed = pd.concat([data_processed, filtered], ignore_index=True)
                forest_vis_data = pd.concat([forest_vis_data, forest_vis_group], ignore_index=True)
                vis_data = forest_vis_data
            data = data_processed

        else:
            outliers = Ifo.fit_predict(data[label].to_numpy().reshape(-1, 1))
            data['outliers'] = outliers

        data_outliers = data[data['outliers'] == -1]
        if remove:
            data = data[data['outliers'] == 1]
        data = data.drop(columns=['outliers'])
        if statistics:
            print('Outlier statistics forest:')
            stats(data=data_outliers, column=label, file='Statistics_Forest.xlsx')

    if confidence:
        if confidence['group']:
            data = data.sort_values(by=confidence['group'], ascending=True).reset_index(drop=True)
            data_processed = pd.DataFrame()
            confidence_vis_data = pd.DataFrame()
            groups = data[confidence['group']].drop_duplicates()
            for index, row in tqdm(groups.iterrows()):
                row = pd.DataFrame(row).transpose()
                filtered = data.merge(row, how='inner', on=confidence['group'])
                filtered[label] = filtered[label].astype(float)

                confidence_vis_group = pd.DataFrame()
                confidence_vis_group['Value'] = filtered[label]
                l = row[confidence['group']].iloc[0, :].apply(str).values
                s = "/".join(l)
                confidence_vis_group['Group'] = s

                mean = filtered[label].mean()
                sigma = filtered[label].std()
                multiplier = confidence['significance']
                filter = (filtered[label] >= mean - multiplier * sigma) & (filtered[label] <= mean + multiplier * sigma)
                filtered['outliers'] = 1
                filtered.loc[~filter, 'outliers'] = -1

                confidence_vis_group['outliers'] = filtered['outliers']
                data_processed = pd.concat([data_processed, filtered], ignore_index=True)
                confidence_vis_data = pd.concat([confidence_vis_data, confidence_vis_group], ignore_index=True)
                vis_data = confidence_vis_data
            data = data_processed

        else:
            mean = data[label].mean()
            sigma = data[label].std()
            multiplier = confidence['significance']

            filter = (data[label] >= mean - multiplier * sigma) & (data[label] <= mean + multiplier * sigma)
            data['outliers'] = 1
            data.loc[~filter, 'outliers'] = -1

        data_outliers = data[data['outliers'] == -1]
        if remove:
            data = data[data['outliers'] == 1]
        data = data.drop(columns=['outliers'])
        if statistics:
            print('Outlier statistics confidence:')
            stats(data=data_outliers, column=label, save=True, file='Statistics_Confidence.xlsx')

    if cap and forest:
        data_outliers = pd.concat([outliers_cap, data_outliers],ignore_index=True)
        if statistics:
            print('Complete outlier statistics:')
            stats(data=data_outliers, column=label, save=True, file='Statistics_Forest_Cap.xlsx')

    if cap and box_plot:
        data_outliers = pd.concat([outliers_cap, data_outliers],ignore_index=True)
        if statistics:
            print('Complete outlier statistics:')
            stats(data=data_outliers, column=label, save=True, file='Statistics_BoxPlot_Cap.xlsx')

    if cap and confidence:
        data_outliers = pd.concat([outliers_cap, data_outliers], ignore_index=True)
        if statistics:
            print('Complete outlier statistics:')
            stats(data=data_outliers, column=label, save=True, file='Statistics_Confidence_Cap.xlsx')

    data_outliers = data_outliers.drop(columns=['outliers'])
    print('Data volume after removing outliers:')
    print(len(data.index))
    #if statistics:
        #stats(data=data, column=label, file='Statistics_Data_All.xlsx')
    return data, data_outliers, vis_data


def visualize_outliers(show_outliers=True, demo=False, vis_data=None, forest=None, box_plot=None, filter=None, sort_by_count=True,
                       title=None, x_axis=None, y_axis=None):

    if not show_outliers:
        vis_data = vis_data[vis_data['outliers'] == 1]

    if filter:
        vis_data = vis_data[vis_data['Group'].isin(filter)]

    if box_plot:
        if box_plot['visualize']:
            sns.set_theme(style="whitegrid")
            counts = vis_data.groupby(["Group"]).size().reset_index(name='count')
            if sort_by_count:
                counts = counts.sort_values(by='count', ascending=False).reset_index(drop=True)
            uniques = counts["Group"]
            chunks = np.array_split(uniques, np.ceil(len(uniques) / box_plot['visualize']))
            n = 1
            for c in chunks:
                chunk_data = isin_filter_sorted(vis_data, 'Group', c)

                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
                ax2 = ax1.twinx()
                ax2.grid(False)
                chunk_data = chunk_data.replace({'outliers': {1: 'in', -1: 'out'}})

                g = sns.boxplot(x="Group", y="Value", data=chunk_data, ax=ax1)
                palette = {"in": 'cornflowerblue', "out": 'red'}
                h1 = sns.histplot(x="Group", hue="outliers", data=chunk_data, multiple='stack',
                                  palette=palette, alpha=0.4,  ax=ax2)
                plt.tight_layout()
                if title:
                    plt.title(title + ' ' + str(n))
                else:
                    plt.title('Outliers Boxplot ' + str(n))
                if x_axis:
                    ax1.set_xlabel(x_axis)
                if y_axis:
                    ax1.set_ylabel(y_axis)
                ax1.set_ylim(ymin=0)
                ax2.set_ylim(ymin=0)
                plt.show()
                n += 1
                if demo:
                    if n == 3:
                        break

    if forest:
        if forest['visualize']:
            sns.set_theme(style="whitegrid")
            counts = vis_data.groupby(["Group"]).size().reset_index(name='count')
            # counts = vis_data.groupby(["Group"]).agg({'Value': [np.ma.count, np.mean, np.std, cv]})
            # counts.columns = ['Group', 'Count', 'Mean', 'Std', 'CV']
            if sort_by_count:
                counts = counts.sort_values(by='count', ascending=False).reset_index(drop=True)

            uniques = counts["Group"]
            chunks = np.array_split(uniques, np.ceil(len(uniques) / forest['visualize']))
            n = 1
            for c in chunks:
                chunk_data = isin_filter_sorted(vis_data, 'Group', c)

                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
                ax2 = ax1.twinx()
                ax2.grid(False)
                chunk_data = chunk_data.replace({'outliers': {1: 'in', -1: 'out'}})

                if forest['density']:
                    chunk_data['Value'] = chunk_data['Value'].astype(float)
                    if len(chunk_data['outliers'].value_counts()) > 1:
                        v1 = sns.violinplot(x="Group", y="Value", hue="outliers",
                                            data=chunk_data, color="grey",
                                            split=False,#True,
                                            scale=forest['scale'],
                                            bw=forest['smoothing_kernel'],
                                            cut=0.1,
                                            ax=ax1, inner=None, legend=None)
                    else:
                        v1 = sns.violinplot(x="Group", y="Value",
                                            data=chunk_data, #palette="Set2",
                                            split=False, #True,
                                            scale=forest['scale'],
                                            bw=forest['smoothing_kernel'],
                                            cut=0.1,
                                            ax=ax1, inner=None, legend=None)
                if forest['scatter']:
                    g1 = sns.scatterplot(x="Group", y="Value", data=chunk_data[chunk_data["outliers"] == 'in'],
                                         marker=forest['scatter'], color='navy', ax=ax1)
                    g2 = sns.scatterplot(x="Group", y="Value", data=chunk_data[chunk_data["outliers"] == 'out'],
                                         marker=forest['scatter'], color='red', ax=ax1)

                palette = {"in": 'cornflowerblue', "out": 'red'}

                h1 = sns.histplot(x="Group", hue="outliers", data=chunk_data, multiple='stack',
                                  palette=palette, alpha=0.4,  ax=ax2)

                #ax1.legend(loc='upper left')
                if forest['density']:
                    v1.legend([], [], frameon=False)
                if title:
                    plt.title(title + ' ' + str(n))
                else:
                    plt.title('Outliers Isolation Forest ' + str(n))
                if x_axis:
                    ax1.set_xlabel(x_axis)
                if y_axis:
                    ax1.set_ylabel(y_axis)
                plt.tight_layout()
                ax1.set_ylim(ymin=0)
                ax2.set_ylim(ymin=0)
                plt.show()

                n += 1
                if demo:
                    if n == 3:
                        break