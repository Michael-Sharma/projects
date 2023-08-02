# Определение пола владельца HTTP cookie по истории активности пользователя в интернете
Необходимо по цифровым следам пользователя (на каких сайтах  он сидел, сколько раз заходил, какое у него устройство и тд) определить его пол. Данные взяты с соревнований MTS ML CUP (https://ods.ai/competitions/mtsmlcup).

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



# Инструкция

## UI Demo
![alt text](intro.gif)

## Основные команды для запуска FastAPI

- Запуск приложения из папки backend, где --reload - указывает на автоматическое обновление при изменении кода

`cd backend`

`uvicorn main:app --host=0.0.0.0 --port=8000 --reload`

Доступ к сервису FastAPI, при условии, что прописали ранее 8000 порт
http://localhost:8000/docs

- Если хотим убить процесс, сначала ищем его и убиваем по id

`lsof -i :8000`

`kill -9 process_id`

- Чтобы сразу убить все процессы

`for pid in $(ps -ef | grep "uvicorn main:app" | awk '{print $2}'); do kill -9 $pid; done`
___

## Основные команды для запуска Streamlit

Обязательно запускаем frontend чась проекта, только после запуска backend  в отдельном терминале

- Команда в отдельном теминале для запуска приложения Streamlit

`cd frontend`

`streamlit run main.py`
И ваше приложение будет доступно по адресу http://localhost:8501 

Если хотите запустить по конкретному порту, например 8080, то:

`streamlit run main.py --server.port 8080`

Убить процессы:

`for pid in $(ps -ef | grep "streamlit run" | awk '{print $2}'); do kill -9 $pid; done`
___

## Configuration file

Важный нюанс! Если запускаете не в контейнере ваш проект, то необходимо поменять в конфигурационном файле endpoints:

```yaml
endpoints:
  train: 'http://localhost:8000/train'
  prediction_input: 'http://localhost:8000/predict_input'
  prediction_from_file: 'http://localhost:8000/predict'
```

Если запускаете в контенере, то вместо localhost должно быть название сервиса или контейнера
```yaml
endpoints:
  train: 'http://fastapi:8000/train'
  prediction_input: 'http://fastapi:8000/predict_input'
  prediction_from_file: 'http://fastapi:8000/predict'
```
___

## Основные команды Docker на примере backend

- Если хотим запустить отдельно образ backend из директории mlops

`docker build -t fastapi:ver1 backend -f backend/Dockerfile`

- Если хотим запустить образ из папки backend

`cd backend`

`docker build -t fastapi:ver1 .`

- Запуск и **создание** нового контейнера из образа fastapi с названием fastapi_run в автономном режиме с использованием портов

`docker run -p 8000:8000 -d --name fastapi_run fastapi:ver1`

- Остановить контейнер

`docker stop fastapi_run`

- Запустить существующий контейнер

`docker start fastapi_run`

- Удалить контейнер

`docker rm fastapi_run`

- При изменении кода необходимо неоьходимо сначала остановить (также можно удалить созданные контейнеры, связнные с образом)

`docker stop fastapi_run`

`docker rm fastapi_run`

- Далее создаем новый образ с новым тэгом

`docker build -t fastapi:ver2 backend -f backend/Dockerfile`

- Далее создаем новый контейнер по новому образу

`docker run -p 8000:8000 -d --name fastapi_run fastapi:ver2`
___

## Docker Compose

- Сборка сервисов из образов внутри backend/frontend и запуск контейнеров в автономном режиме

`docker compose up -d`

- Если сделали изменения в коде необходимо заново пересобрать образы

`docker compose up -d --build`

- Оставнока сервисов (если хотите например сделать изменения в коде)

`docker compose stop`

- Удалить **остановленные** контейнеры

`docker compose rm`


___
## Folders
- `/backend` - Папка с проектом FastAPI
- `/frontend` - Папка с проектом Streamlit
- `/config` - Папка, содержащая конфигурационный файл
- `/data` - Папка, содержащая исходные данные, обработанные данные и уникальные значения определенных признаков в формате JSON
- `/demo` - Папка, содержащая демо работы сервиса в Streamlit UI в формате gif
- `/models` - Папка, содержащая сохраненную модель после тренировки, а также объект study (Optuna)
- `/notebooks` - Папка, содержащая jupyter ноутбуки с предварительным анализом данных
- `/report` - Папка, содержащая информацию о лучших параметрах и метриках после обучения
