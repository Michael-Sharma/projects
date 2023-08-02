"""
Получение предсказания на основе обученной модели
"""

import os
import yaml
import joblib
import pandas as pd
from ..transform.transform import pipeline_preprocess


def pipeline_evaluate(
    config_path, dataset: pd.DataFrame = None
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param dataset: датасет
    :param config_path: путь до конфигурационного файла
    :return: предсказания
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    dataset = pipeline_preprocess(data=dataset, **preprocessing_config)

    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(dataset).tolist()

    return prediction


