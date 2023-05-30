import re
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from ast import literal_eval
from tqdm import tqdm
from fuzzywuzzy import fuzz
from data_process import DATASET_PATH, WORK_PATH, PARQUET_ENGINE
from data_process import DataPreprocess, DataTransform, TextLemmatization
from print_time import print_time, print_msg

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

__import__("warnings").filterwarnings('ignore')

PREDICTIONS_DIR = WORK_PATH.joinpath('predictions')
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_LOGS = Path(r'D:\python-txt\LCT_OZON\scores_local.logs')
if not MODELS_LOGS.is_file():
    MODELS_LOGS = WORK_PATH.joinpath('scores_local.logs')

train_pairs = DATASET_PATH.joinpath('train_pairs.parquet')
test_pairs = DATASET_PATH.joinpath('test_pairs_wo_target.parquet')
file_train = DATASET_PATH.joinpath('train_data.parquet')
file_test = DATASET_PATH.joinpath('test_data.parquet')


def predict_auc_macro(df: pd.DataFrame, prec_level: float = 0.75,
                      cat_column: str = "cat3_grouped") -> float:
    y_true = df["target"]
    y_pred = df["scores"]
    categories = df[cat_column]

    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    for i, category in enumerate(unique_cats):
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred.iloc[cat_idx]
        y_true_cat = y_true.iloc[cat_idx]

        y, x, thr = precision_recall_curve(y_true_cat, y_pred_cat)
        gt_prec_level_idx = np.where(y >= prec_level)[0]

        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
                weights.append(counts[i] / len(categories))
        except ValueError as err:
            pr_aucs.append(0)
            weights.append(0)
    return np.average(pr_aucs, weights=weights)


def read_train_test_sample(rebuilding_pairs=False, rows=500):
    file_preprocess = WORK_PATH.joinpath('preprocess_sample.pkl')
    dts_object = WORK_PATH.joinpath('dts_object_sample.pkl')
    files = (WORK_PATH.joinpath('test_smp.pkl'), WORK_PATH.joinpath('train_smp.pkl'))
    start_time = print_msg('Готовлю небольшую выборку с парами')

    # читаем списки товаров в один ДФ
    read_sample = False
    tov = pd.concat([pd.read_parquet(file, engine=PARQUET_ENGINE)
                     for file in (file_train, file_test)[read_sample:]],
                    ignore_index=True)
    tov.drop_duplicates(['variantid'], inplace=True)

    dpr = DataPreprocess()

    train = pd.read_parquet(train_pairs, engine=PARQUET_ENGINE)
    tov_train, train = dpr.make_sample(tov, train, rows=rows)
    tov_train = dpr.fit_transform(tov_train)
    test = pd.read_parquet(test_pairs, engine=PARQUET_ENGINE)
    tov_test, test = dpr.make_sample(tov, test, rows=rows // 2)
    tov_test = dpr.fit_transform(tov_test)

    tov = pd.concat([tov_train, tov_test])

    # tov.to_pickle(file_preprocess)
    # tov.to_csv(file_preprocess.with_suffix('.csv'), sep=';')

    cat_columns = ['cat3', 'categories', 'color']
    dts = DataTransform(category_columns=cat_columns)
    dts.exclude_columns = ['name', 'color', 'pic_emb', 'ch_att_map',
                           'cam',
                           'cat',
                           'name_pic',
                           # 'main_pic',
                           # 'name_bert',
                           'cos_pics', 'cos_values', 'cos_idxs',
                           'name_norm1', 'name_stem1', 'name_lemm1',
                           'name_norm2', 'name_stem2', 'name_lemm2',
                           'cam_norm1', 'cam_stem1', 'cam_lemm1',
                           'cam_norm2', 'cam_stem2', 'cam_lemm2',
                           'name_stem_lemm1', 'name_stem_lemm2',
                           ]

    for idx, dfp in enumerate((train, test)):
        # print(dfp.info())

        file = files[idx]
        dts.preprocess_path_file = WORK_PATH.joinpath(f'preprocess_{file.name}')

        dfp = dts.fit_transform(tov, dfp,
                                rebuilding_pairs=rebuilding_pairs,
                                preprocess_to_csv=True,
                                # expand_columns=['name_bert', 'epic']
                                )

        print('tov.columns:', dfp.columns.to_list())
        print('numeric_columns:', dts.numeric_columns)
        print('category_columns:', dts.category_columns)
        print('exclude_columns:', dts.exclude_columns)
        dfp.to_pickle(files[idx])
        with open(dts_object, "wb") as file_object:
            pickle.dump(dts, file_object)

    train = pd.read_pickle(files[0])
    test = pd.read_pickle(files[1])
    with open(dts_object, "rb") as file_object:
        dts = pickle.load(file_object)

    print_time(start_time)
    return train, test, dts


def read_train_test(rebuilding_pairs=False, remake_train_test_files=False):
    """
    Чтение подготовленных данных, если файлы отсутствуют - выполняется предобработка
    :param rebuilding_pairs: полностью перестроить ДФ с парами train и test
    :param remake_train_test_files: пересобрать ДФ с парами train и test
    :return:
    """
    file_preprocess = DATASET_PATH.joinpath('preprocess_data.pkl')
    dts_object = DATASET_PATH.joinpath('dts_object.pkl')

    need_make_files = True
    if file_preprocess.is_file():
        # если подготовленных файлов нет - читаем предобработанный ДФ и готовим их
        if rebuilding_pairs or not all(file.with_suffix('.pkl').is_file()
                                       for file in (train_pairs, test_pairs)):
            start_time = print_msg('Читаю предобработанный ДФ')
            tov = pd.read_pickle(file_preprocess)
        else:
            need_make_files = False
    else:
        start_time = print_msg('Готовлю датасет с товарами')
        # читаем списки товаров в один ДФ
        read_sample = False
        tov = pd.concat([pd.read_parquet(file, engine=PARQUET_ENGINE)
                         for file in (file_train, file_test)[read_sample:]],
                        ignore_index=True)

        tov.drop_duplicates(['variantid'], inplace=True)

        dpr = DataPreprocess()
        tov = dpr.fit_transform(tov)

        print('Сохраняю предобработанный ДФ')
        tov.to_pickle(WORK_PATH.joinpath('preprocess_data.pkl'))
        print_time(start_time)

        print(tov.info())
        print(tov.columns.to_list())

    if need_make_files or remake_train_test_files:

        if remake_train_test_files:
            tov = None
            rebuilding_pairs = False

        cat_columns = ['cat3', 'categories', 'color']
        dts = DataTransform(category_columns=cat_columns)
        dts.exclude_columns = ['name', 'color', 'pic_emb', 'ch_att_map',
                               'cam',
                               'cat',
                               'name_pic',
                               # 'main_pic',
                               # 'name_bert',
                               'cos_pics', 'cos_values', 'cos_idxs',
                               'name_norm1', 'name_stem1', 'name_lemm1',
                               'name_norm2', 'name_stem2', 'name_lemm2',
                               'cam_norm1', 'cam_stem1', 'cam_lemm1',
                               'cam_norm2', 'cam_stem2', 'cam_lemm2',
                               'name_stem_lemm1', 'name_stem_lemm2',
                               ]

        for idx, file in enumerate((train_pairs, test_pairs)):
            if (rebuilding_pairs or remake_train_test_files
                    or not file.with_suffix('.pkl').is_file()):
                start_time = print_msg(f"Готовлю {('тренировочный', 'тестовый')[idx]} "
                                       f"датасет")
                dfp = pd.read_parquet(file, engine=PARQUET_ENGINE)

                print(dfp.info())

                dts.preprocess_path_file = WORK_PATH.joinpath(
                    f'preprocess_{file.name}').with_suffix('.pkl')

                dfp = dts.fit(tov, dfp, rebuilding_pairs=rebuilding_pairs)

                dfp = dts.transform(tov, dfp)

                print('tov.columns:', dfp.columns.to_list())
                print('numeric_columns:', dts.numeric_columns)
                print('category_columns:', dts.category_columns)
                print('exclude_columns:', dts.exclude_columns)
                dfp.to_pickle(WORK_PATH.joinpath(file.name).with_suffix('.pkl'))
                print(dfp.info())

                with open(dts_object, "wb") as file_object:
                    pickle.dump(dts, file_object)

                print_time(start_time)

        work_path = WORK_PATH
    else:
        work_path = DATASET_PATH

    start_time = print_msg('Читаю датасеты с парами товаров')

    train = pd.read_pickle(work_path.joinpath(train_pairs.name).with_suffix('.pkl'))
    test = pd.read_pickle(work_path.joinpath(test_pairs.name).with_suffix('.pkl'))
    with open(dts_object, "rb") as file_object:
        dts = pickle.load(file_object)

    print_time(start_time)
    return train, test, dts


def get_max_num(file_logs=None):
    """Получение максимального номера итерации обучения моделей
    :param file_logs: имя лог-файла с полным путем
    :return: максимальный номер
    """
    if file_logs is None:
        file_logs = MODELS_LOGS

    if not file_logs.is_file():
        with open(file_logs, mode='a') as log:
            log.write('num;mdl;roc_auc;acc_train;acc_valid;sc_train;score;WF1;'
                      'model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        df = pd.read_csv(file_logs, sep=';')
        df.num = df.index + 1
        max_num = df.num.max()
    return max_num


def predict_train_valid(model, datasets, label_enc=None, max_num=0):
    """Расчет метрик для модели: accuracy на трейне, на валидации, на всем трейне, roc_auc
    и взвешенная F1-мера на валидации
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param label_enc: используемый label_encоder для target'а
    :param max_num: максимальный порядковый номер обучения моделей
    :return: accuracy на трейне, на валидации, на всем трейне, roc_auc и взвешенная F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, trn = datasets
    valid_pred = model.predict(X_valid)
    train_pred = model.predict(X_train)
    train_full = model.predict(train)

    predict_train = model.predict_proba(X_train)[:, 1]
    # выберем строки по индексам обучающей выборки
    df = trn.loc[X_train.index]
    df["scores"] = predict_train
    score_train = predict_auc_macro(df)

    predict_proba = model.predict_proba(X_valid)[:, 1]
    # выберем строки по индексам валидационной выборки
    df = trn.loc[X_valid.index]
    df["scores"] = predict_proba
    score_valid = predict_auc_macro(df)

    precision, recall, thrs = precision_recall_curve(df["target"], df["scores"])
    pr_auc = auc(recall, precision)
    fig, ax1 = plt.subplots(1, figsize=(15, 7))
    ax1.plot(recall, precision)
    ax1.axhline(y=0.75, color='grey', linestyle='-')
    # сохранение картинки
    file_to_save = f'pr_auc_{max_num:03}_local.png'
    plt.savefig(PREDICTIONS_DIR.joinpath(file_to_save))
    # plt.show()

    f1w = f1_score(y_valid, valid_pred)
    acc_valid = accuracy_score(y_valid, valid_pred)
    acc_train = accuracy_score(y_train, train_pred)
    acc_full = accuracy_score(target, train_full)
    roc_auc = roc_auc_score(y_valid, predict_proba)

    print(f'Score auc_macro = train:{score_train:.6f} valid:{score_valid:.6f}')
    print(f'Accuracy train:{acc_train} valid:{acc_valid} roc_auc:{roc_auc} F1:{f1w:.6f}')

    return acc_train, acc_valid, roc_auc, f1w, score_train, score_valid


def predict_test(idx_fold, model, datasets, max_num=0, submit_prefix='lg_', label_enc=None,
                 save_predict_proba=True):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: accuracy на трейне, на валидации, на всем трейне, roc_auc
    и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param label_enc: используемый label_encоder для target'а
    :param save_predict_proba: сохранять файл с вероятностями предсказаний
    :return: accuracy на трейне, на валидации, на всем трейне, roc_auc и взвешенная F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, trn = datasets
    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''
    columns_no_to_model = ['target', 'variantid1', 'variantid2']
    predictions = model.predict(test_df.drop(columns_no_to_model[1:], axis=1))
    predict_train = model.predict(train)

    if label_enc:
        predictions = label_enc.inverse_transform(predictions)
        predict_train = label_enc.inverse_transform(predict_train)

    # печать размерности предсказаний и списка меток классов
    classes = model.classes_.tolist()
    print('predict_proba.shape:', predictions.shape, 'classes:', classes)

    predict_proba = model.predict_proba(test_df.drop(columns_no_to_model[1:], axis=1))
    train_proba = model.predict_proba(train)

    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}_local.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    file_proba_csv = PREDICTIONS_DIR.joinpath(submit_csv.replace('submit_', 'proba_'))
    file_train_csv = PREDICTIONS_DIR.joinpath(submit_csv.replace('submit_', 'train_'))

    # Сохранение предсказаний в файл
    submit = test_df[columns_no_to_model[1:] + ['cat3_grouped']]
    submit['target'] = predict_proba[:, 1]
    submit.to_csv(file_submit_csv, index=False)
    if save_predict_proba:
        train_sp = trn[columns_no_to_model]
        train_sp['pred_target'] = predict_train
        train_sp.to_csv(file_train_csv, index=False)

        try:
            train_sp[classes] = train_proba
            train_sp.to_csv(file_train_csv, index=False)
        except:
            pass

    acc_tr, acc_val, roc_auc, f1w, sc_tr, sc_val = predict_train_valid(model,
                                                                       datasets,
                                                                       label_enc=label_enc,
                                                                       max_num=max_num)
    return acc_tr, acc_val, roc_auc, f1w, sc_tr, sc_val


def add_features_to_preprocessed_file():
    """Добавление фич в предобработанный файл с товаром"""

    def calc_frset(a, b):
        set1, set2 = map(set, (a, b))
        return 1 - len(set1.symmetric_difference(set2)) / len(set1.union(set2))

    file_name = 'preprocess_data.pkl'
    file_preprocess = DATASET_PATH.joinpath(file_name)

    start_time = print_msg(f'Читаю предобработанный ДФ: {file_name}')
    if file_preprocess.is_file():
        tov = pd.read_pickle(file_preprocess)
        print(tov.columns.to_list())
    else:
        print('Ошибка!!! Предобработанный файл не обнаружен: запустите read_train_test()')
        return

    print_time(start_time)

    dpr = DataPreprocess()

    tqdm.pandas()

    for col in ('cat3_x', 'cat3_y'):
        if col in tov.columns:
            tov.drop(col, axis=1, inplace=True)

    print('Сохраняю предобработанный ДФ')
    tov.to_pickle(WORK_PATH.joinpath(file_name))
    print_time(start_time)

    tqdm.pandas()

    dts = DataTransform()

    for idx, file in enumerate((train_pairs, test_pairs)):
        start_time = print_msg(f"Готовлю {('тренировочный', 'тестовый')[idx]} "
                               f"датасет")

        dts.preprocess_path_file = WORK_PATH.joinpath(
            f'preprocess_{file.name}').with_suffix('.pkl')

        # читаем предобработанный файл
        pair = dts.fit(None, None)

        # обработка только колонок с характеристиками товара
        for col in ('cat3_x', 'cat3_y'):
            if col in pair.columns:
                pair.drop(col, axis=1, inplace=True)

        print_time(start_time)
        start_time = print_msg(f'Сохраняю {dts.preprocess_path_file.name}')
        pair.to_pickle(dts.preprocess_path_file)
        print_time(start_time)


def add_features_to_pairs_preprocessed():
    """Добавление фич в предобработанный файл с парами товаров"""

    # file_name = 'preprocess_data.pkl'
    # file_preprocess = DATASET_PATH.joinpath(file_name)
    #
    # start_time = print_msg(f'Читаю предобработанный ДФ: {file_name}')
    # if file_preprocess.is_file():
    #     tov = pd.read_pickle(file_preprocess)
    #     print(tov.columns.to_list())
    # else:
    #     print('Ошибка!!! Предобработанный файл не обнаружен: запустите read_train_test()')
    #     return
    #
    # print_time(start_time)

    tqdm.pandas()

    dts = DataTransform()

    for idx, file in enumerate((train_pairs, test_pairs)):
        start_time = print_msg(f"Готовлю {('тренировочный', 'тестовый')[idx]} "
                               f"датасет")

        dts.preprocess_path_file = WORK_PATH.joinpath(
            f'preprocess_{file.name}').with_suffix('.pkl')

        # читаем предобработанный файл
        pair = dts.fit(None, None)

        for col in ('cat3_x', 'cat3_y'):
            if col in pair.columns:
                pair.drop(col, axis=1, inplace=True)

        pair['cat31'] = pair['cat1'].apply(lambda x: x['3'])
        pair['cat32'] = pair['cat2'].apply(lambda x: x['3'])

        print_time(start_time)
        start_time = print_msg(f'Сохраняю {dts.preprocess_path_file.name}')
        pair.to_pickle(dts.preprocess_path_file)
        print_time(start_time)


if __name__ == '__main__':
    pass

    # # 1
    # add_features_to_preprocessed_file()
    # # 2
    # add_features_to_pairs_preprocessed()
    # # 3
    # train_df, test_df, dts = read_train_test(rebuilding_pairs=False,
    #                                          remake_train_test_files=True)

    # # перестройка всех данных
    # _, _, dts_obj = read_train_test(rebuilding_pairs=True)
