# ЛИДЕРЫ ЦИФРОВОЙ ТРАНСФОРМАЦИИ 2023
ЗАДАЧА 5.
ПОИСК ОДИНАКОВЫХ ТОВАРОВ НА МАРКЕТПЛЕЙСЕ
https://leaders2023.innoagency.ru/task_5.html

# Порядок работы

1. В файле data_process.py задать путь к данным DATASET_PATH и WORK_PATH (сабмиты будут складываться в каталог WORK_PATH\predictions).
2. Для определения языка в каталог DATASET_PATH положить файл lid.176.bin
3. В файле requirements.txt необходимые модули для работы.
4. Для подготовки данных запустить скрипт 0_process_data.py (Для комфортной работы ОЗУ не менее 32Г).
5. Классификатор:
  - Локальная версия CatBoostClassifier: ozon_catboost.py 
  - Локальная версия LGBMClassifier: ozon_lightgbm.py (с ним результат пониже)
  - Запустить Jupyter Notebook ozon-catboost.ipynb https://www.kaggle.com/code/awesomesp68/ozon-catboost-2
