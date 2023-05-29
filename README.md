# ЛИДЕРЫ ЦИФРОВОЙ ТРАНСФОРМАЦИИ 2023
ЗАДАЧА 5.
ПОИСК ОДИНАКОВЫХ ТОВАРОВ НА МАРКЕТПЛЕЙСЕ
https://leaders2023.innoagency.ru/task_5.html

# Порядок работы

1. В файле data_process.py задать путь к данным DATASET_PATH и WORK_PATH (сабмиты будут складываться в каталог WORK_PATH\predictions).
2. Для подготовки данных запустить скрипт 0_process_data.py (Для комфортной работы ОЗУ не менее 32Г).
- Запустить локально классификатор CatBoostClassifier: ozon_catboost.py или LGBMClassifier: ozon_lightgbm.py (с ним результат хуже).
- Запустить Jupyter Notebook ozon-catboost.ipynb на www.kaggle.com 
