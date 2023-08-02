"""
Backend часть проекта
"""

import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class IsMale(BaseModel):
    """
    Признаки для получения результатов модели
    """
    region_name: str
    city_name: str
    cpe_manufacturer_name: str
    cpe_model_name: str
    url_host: str
    cpe_model_os_type: str
    price: float
    part_of_day: str
    request_cnt: int


# @app.get("/hello")
# def welcome():
#     """
#     Hello
#     :return: None
#     """
#     return {'message': 'Hello Data Scientist!'}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict_input")
def prediction_input(user: IsMale):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            user.region_name,
            user.city_name,
            user.cpe_manufacturer_name,
            user.cpe_model_name,
            user.url_host,
            user.cpe_model_os_type,
            user.price,
            user.part_of_day,
            user.request_cnt
        ]
    ]

    cols = [
        "region_name",
        "city_name",
        "cpe_manufacturer_name",
        "cpe_model_name",
        "url_host",
        "cpe_model_os_type",
        "price",
        "part_of_day",
        "request_cnt"
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]

    if predictions == 1:
        result = "The user is male"
    elif predictions == 0:
        result = "The user is female"
    else:
        result = "Error result"

    return result


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)

