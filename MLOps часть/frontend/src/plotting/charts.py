"""
Отрисовка графиков
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def target_distribution(
    data: pd.DataFrame, title: str, xlabel: str, **kwargs
)-> matplotlib.figure.Figure:
    """
    Отрисовка графика barplot для таргета
    :param data: датасет
    :param title: название графика
    :param xlabel: подпись оси OX
    :return: поле рисунка
    """
    target_values = (data[kwargs["target_column"]]
                     .value_counts(normalize=True)
                     .mul(100)
                     .rename('percent')
                     .reset_index())

    fig = plt.figure(figsize=(10, 7))

    ax = sns.barplot(x='index',
                     y='percent',
                     data=target_values,
                     palette='mako')

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=14)

    plt.title(title, fontsize=20)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Проценты', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    return fig


def barplot_group(
    data: pd.DataFrame, col_main: str, col_agg: str,  title: str, xlabel: str, **kwargs
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: датасет
    :param col_main: признак для анализа в разрезе таргета
    :param col_agg: вспомогательная колонка для построения сводной таблицы (может быть любая колонка без пропусков
                                                                            != col_main)
    :param title: название графика
    :param xlabel: подпись оси OX
    :return: поле рисунка
    """
    pivot = data.pivot_table(index=[kwargs["target_column"], col_main],
                             values=col_agg,
                             aggfunc='count').reset_index()
    pivot_1 = pivot.groupby(kwargs["target_column"]).sum(col_agg).reset_index().rename(columns={col_agg: 'sum_target'})
    pivot = pivot.merge(pivot_1, on=kwargs["target_column"])
    pivot['percent'] = (pivot[col_agg] / pivot['sum_target']) * 100

    pivot = pivot.sort_values('percent', ascending=False)

    fig = plt.figure(figsize=(12, 7))

    ax = sns.barplot(x=col_main,
                     y='percent',
                     data=pivot,
                     palette='mako',
                     hue=kwargs["target_column"])

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=14)

    plt.title(title, fontsize=20)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Проценты', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    return fig


def kdeplotting(
    data: pd.DataFrame, column: str, title: str, xlabel: str, **kwargs
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика kdeplot
    :param data: датасет
    :param column: ось OX
    :param title: название графика
    :param xlabel: подпись оси OX
    :return: поле рисунка
    """

    fig = plt.figure(figsize=(12, 6))

    sns.kdeplot(
        data=data, x=column, hue=kwargs["target_column"], palette="mako", common_norm=False
    )
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    return fig



