"""
Предобработка данных
"""

import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns+[target_column], axis=1, errors="ignore"
    )
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file, ensure_ascii=False)


def pipeline_preprocess(data: pd.DataFrame, flg_evaluate: bool = True, **kwargs):
    """
    Пайплайн по предобработке данных
    :param data: датасет
    :param flg_evaluate: флаг для evaluate
    :return: предобработанный датасет
    """
    if flg_evaluate:
        # transform types
        data = transform_types(data=data, change_type_columns=kwargs["change_type_columns"])
    else:
        save_unique_train_data(
            data=data,
            drop_columns=kwargs["drop_for_unique"],
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )
        # drop columns
        data = data.drop(kwargs["drop_columns"], axis=1, errors="ignore")
        # transform types
        data = transform_types(data=data, change_type_columns=kwargs["change_type_columns"])

    return data
