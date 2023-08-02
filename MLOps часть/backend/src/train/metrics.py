"""
Получение метрик
"""
import json

import yaml
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
import pandas as pd


def create_dict_metrics(
    y_test: pd.Series, y_predict: pd.Series, y_probability: pd.Series
) -> dict:
    """
    Получение словаря с метриками для задачи классификации и запись в словарь
    :param y_test: реальные данные
    :param y_predict: предсказанные значения
    :param y_probability: предсказанные вероятности
    :return: словарь с метриками
    """
    dict_metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probability[:, 1]), 3),
        "precision": round(precision_score(y_test, y_predict), 3),
        "recall": round(recall_score(y_test, y_predict), 3),
        "f1": round(f1_score(y_test, y_predict), 3),
        "logloss": round(log_loss(y_test, y_probability), 3)
    }

    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(y_test=data_y,
                                         y_predict=model.predict(data_x),
                                         y_probability=model.predict_proba(data_x),
                                         )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
