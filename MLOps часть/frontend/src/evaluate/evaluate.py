"""
Отрисовка слайдеров, кнопок полей для ввода данных
с дальнейшим получением предсказания на основании введенных значений
"""

import json
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, dataset: pd.DataFrame, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений категориальных признаков
    :param dataset: датасет
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    region_name = st.sidebar.selectbox("Region", (unique_df["region_name"]))
    city_name = st.sidebar.selectbox("City", (unique_df["city_name"]))
    cpe_manufacturer_name = st.sidebar.selectbox("Brand", (unique_df["cpe_manufacturer_name"]))
    cpe_model_name = st.sidebar.selectbox("Model", (unique_df["cpe_model_name"]))
    url_host = st.sidebar.text_input("URL")
    cpe_model_os_type = st.sidebar.selectbox("OS_Type", (unique_df["cpe_model_os_type"]))
    price = st.sidebar.slider(
        "Price",
        min_value=dataset['price'].min(),
        max_value=dataset['price'].max(),
        value=float(dataset['price'][0])
    )
    part_of_day = st.sidebar.selectbox("Part_of_day", (unique_df["part_of_day"]))
    request_cnt = st.sidebar.slider(
        "Requests",
        min_value=int(dataset['request_cnt'].min()),
        max_value=int(dataset['request_cnt'].max()),
        value=int(dataset['request_cnt'][0]),
        step=1
    )

    dict_data = {
        "region_name": region_name,
        "city_name": city_name,
        "cpe_manufacturer_name": cpe_manufacturer_name,
        "cpe_model_name": cpe_model_name,
        "url_host": url_host,
        "cpe_model_os_type": cpe_model_os_type,
        "price": price,
        "part_of_day": part_of_day,
        "request_cnt": request_cnt
    }

    st.write(
        f"""### Данные посетителя:\n
        1) Region: {dict_data['region_name']}
        2) City: {dict_data['city_name']}
        3) Brand: {dict_data['cpe_manufacturer_name']}
        4) Model: {dict_data['cpe_model_name']}
        5) URL: {dict_data['url_host']}
        6) OS_Type: {dict_data['cpe_model_os_type']}
        7) Price: {dict_data['price']}
        8) Part_of_day: {dict_data['part_of_day']}
        9) Requests: {dict_data['request_cnt']}
        """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output}")
        st.success("Success!")


