"""
Разделение на train/test
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(dataset: pd.DataFrame, **kwargs):
    """
    Разделение данных на train/val/test с последующим сохранением
    :param dataset: датасет
    :return: train/val/test датасеты
    """
    df_train, df_test = train_test_split(dataset,
                                         stratify=dataset[kwargs["target_column"]],
                                         test_size=kwargs["test_size"],
                                         random_state=kwargs["random_state"],)

    df_train_, df_val = train_test_split(df_train,
                                         stratify=df_train[kwargs["target_column"]],
                                         test_size=kwargs["val_size"],
                                         random_state=kwargs["random_state"],)

    df_train_.to_csv(kwargs["train_path_proc"], index=False)
    df_test.to_csv(kwargs["test_path_proc"], index=False)

    return df_train_, df_val,  df_test


def get_train_test_data(
    data_train: pd.DataFrame, data_val: pd.DataFrame, data_test: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Получение train/val/test данных разбитых по отдельности на объект-признаки и целевую переменную
    :param data_train: train датасет
    :param data_val: val датасет
    :param data_test: test датасет
    :param target: название целевой переменной
    :return: набор данных train/test
    """
    x_train, x_val, x_test = (
        data_train.drop(target, axis=1),
        data_val.drop(target, axis=1),
        data_test.drop(target, axis=1)
    )
    y_train, y_val, y_test = (
        data_train[target],
        data_val[target],
        data_test[target]
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
