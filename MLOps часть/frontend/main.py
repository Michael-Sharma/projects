"""
Frontend часть проекта
"""

import os

import pandas as pd
import yaml
import streamlit as st
from src.data.get_data import get_dataset
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input
from src.plotting.charts import target_distribution, barplot_group, kdeplotting

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "logo.jpg",
        width=700,

    )
    st.markdown("# Описание проекта")
    st.title("MLOps project:  Определение пола владельца HTTP cookie по истории активности пользователя в интернете")
    st.write(
        """
        Необходимо по цифровым следам пользователя (на каких сайтах он сидел, сколько раз заходил,
        какое у него устройство и тд) определить его пол. Данные взяты с соревнований MTS ML CUP
        (https://ods.ai/competitions/mtsmlcup)."""
    )
    st.markdown(
        """
            Описание колонок файла с данными:
            
                - 'region_name' – Регион
                - 'city_name' – Населенный пункт
                - 'cpe_manufacturer_name' – Производитель устройства
                - 'cpe_model_name' – Модель устройства
                - 'url_host' – Домен, с которого пришел рекламный запрос
                - 'cpe_model_os_type' – Операционка на устройстве
                - 'price' – Оценка цены устройства
                - 'date' – Дата
                - 'part_of_day' – Время дня (утро, вечер, итд)
                - 'request_cnt' – Число запросов одного пользователя за время дня (поле part_of_day)
                - 'user_id' – ID пользователя

            Таргет:
                
                - 'is_male' – Признак пользователя : мужчина (1-Да, 0-Нет)
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]

    # load and write dataset
    data = get_dataset(dataset_path=preprocessing_config["data_path"])
    st.write(data.head())

    target_dist = st.sidebar.checkbox("Распределение таргета")
    part_of_day = st.sidebar.checkbox("Время дня")
    manufacturer = st.sidebar.checkbox("Производитель устройства")
    os_type = st.sidebar.checkbox("Операционная система на устройстве")
    price = st.sidebar.checkbox("Цена устройства")

    if target_dist:
        st.pyplot(
            target_distribution(data=data,
                                title='Процентное распределение пользователей по полу',
                                xlabel='Пол',
                                **preprocessing_config)
        )
        st.markdown("**Вывод:** дисбаланса классов не наблюдается")
    if part_of_day:
        st.pyplot(
            barplot_group(data=data,
                          col_main='part_of_day',
                          col_agg='user_id',
                          title='Доли пользователей в разрезе пола \n в зависимости от времени дня',
                          xlabel='Время дня',
                          **preprocessing_config
                          )
        )
        st.markdown("**Вывод:** Чаще всего пользователи выходят в сеть днем, чуть реже вечером и утром."
                    " Ночью, как правило, люди спят, отсюда и небольшое число пользователей, выходивших "
                    "в сеть в сравнении с остальными временами дня. Стоит отметить, что днем и вечером "
                    "чаще выходят в сеть женщины, а утром и ночью мужчины.")
    if manufacturer:
        st.pyplot(
            barplot_group(data=data,
                          col_main='cpe_manufacturer_name',
                          col_agg='user_id',
                          title='Доли пользователей в разрезе пола \n в зависимости от производителя устройства',
                          xlabel='Производитель устройства',
                          **preprocessing_config
                          )
        )
        st.markdown("**Вывод:** Лидирующим брендом по количеству пользователей является Apple. Дальше идут Samsung, "
                    "Huawei и Xiaomi. Пользователей с устройствами других производителей значительно меньше. При этом "
                    "среди пользователей Apple и Samsung больше женщин, а среди пользователей остальных производителей "
                    "больше мужчин.")
    if os_type:
        st.pyplot(
            barplot_group(data=data,
                          col_main='cpe_model_os_type',
                          col_agg='user_id',
                          title='Доли пользователей в разрезе пола \n в зависимости от операционной'
                                ' системы на устройстве',
                          xlabel='Производитель устройства',
                          **preprocessing_config
                          )
        )
        st.markdown("**Вывод:** Процент пользователей Android больше, чем у iOS (пользователей Apple больше, чем у "
                    "других производителей по отдельности, но не в сумме). При этом Android больше предпочитают "
                    "мужчины, а iOS женщины.")
    if price:
        st.pyplot(
            kdeplotting(data=data,
                        column='price',
                        title='Распределение цены устройства в разрезе пола',
                        xlabel='Цена устройства',
                        **preprocessing_config)
        )
        st.markdown("**Вывод:** характеры распределений цены устройства в разрезе пола практически идентичны")


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model LightGBM")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    preprocessing_config = config["preprocessing"]
    data = get_dataset(dataset_path=preprocessing_config["data_path"])

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(preprocessing_config["unique_values_path"], dataset=data, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()

