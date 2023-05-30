import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import seaborn as sns

import lightgbm as lg
import optuna

from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from data_process import DataTransform

from some_functions import read_train_test, read_train_test_sample
from some_functions import MODELS_LOGS, predict_test, predict_train_valid, get_max_num
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    trial_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "seed": SEED,
        "verbosity": -1,
        # "num_boost_round": 600,
        "num_boost_round": trial.suggest_int("num_boost_round", 500, 650, step=50),
        # "boosting_type": "goss",
        # "boosting_type": trial.suggest_categorical("boosting_type",
        #                                            ["gbdt", "dart", "goss"]),
        "class_weight": None,
        # "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        # "is_unbalance": True,
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        # "learning_rate": 0.01,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, step=0.005),
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, step=0.01),

        # "depth": trial.suggest_int("depth", 2, 12),
        # "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
    gbm = lg.train(trial_params, pool_train, valid_sets=[pool_valid],
                   callbacks=[lg.early_stopping(50), pruning_callback])

    pred_valid = gbm.predict(X_valid)
    pred_labels = np.argmax(pred_valid, axis=1)
    accuracy = accuracy_score(y_valid, pred_labels)
    return accuracy


file_logs = MODELS_LOGS
max_num = get_max_num(file_logs)

start_time = print_msg('Обучение lightGBM классификатор...')

data_cls = DataTransform()

# # небольшая выборка для тестов
# train_df, test_df, dts = read_train_test_sample(rebuilding_pairs=True)

# чтение подготовленного датасета
train_df, test_df, dts = read_train_test(rebuilding_pairs=False)

# print(train_df.shape, test_df.shape, set(train_df.columns) - set(test_df.columns),
#       train_df.columns.to_list().index('cat3'))
# print(train_df.columns.to_list())
# print(test_df.columns.to_list())

cam_atrs = sorted(filter(lambda x: re.fullmatch(r'atr_\d+', x), train_df.columns))
fuz_atts = sorted(filter(lambda x: re.fullmatch(r'fuz_atr_\d+', x), train_df.columns))
comp_atrs = sorted(filter(lambda x: re.fullmatch(r'comp_atr_\d+', x), train_df.columns))
keys_atrs = sorted(filter(lambda x: re.fullmatch(r'keys_\d+', x), train_df.columns))

# количество характеристик товара, которые будем использовать: задавать четное число
numbers_cam_atrs = 0  # 99 * 2
numbers_fuz_atrs = 200  # 99 * 2
numbers_comp_atrs = 200  # 99 * 2
numbers_keys_atrs = 200  # 99 * 2
# перезапишем словарь из класса dts в класс data_cls
dts.exclude_columns.extend(['pic_emb_mean', 'cam_lemm', 'cam_lm_len', 'cos_pics',
                            'cos_pic_emb_mean'])
dts.exclude_columns.extend(cam_atrs[numbers_cam_atrs:])
dts.exclude_columns.extend(fuz_atts[numbers_fuz_atrs:])
dts.exclude_columns.extend(comp_atrs[numbers_comp_atrs:])
dts.exclude_columns.extend(keys_atrs[numbers_keys_atrs:])

dts.exclude_columns.extend(['fuz_cam_lemm', 'fract_cam_lm_len',
                            'name_lm_len', 'name_stem_lemm',
                            'fuz_name_lemm',
                            'fract_name_lm_len',
                            ])

# if 'cat3' in dts.exclude_columns:
#     dts.exclude_columns.remove('cat3')

# удаление категорийных колонок, если они есть в исключенных
cat_columns = dts.__dict__['category_columns']
cat_columns = [col for col in cat_columns if col not in dts.exclude_columns]
dts.__dict__['category_columns'] = cat_columns

data_cls.__dict__ = dts.__dict__.copy()
data_cls.category_columns.append('cat3_grouped')
# data_cls.category_columns.append('cat3')
data_cls.category_columns.extend(cam_atrs[:numbers_cam_atrs])
data_cls.category_columns.extend(keys_atrs[:numbers_keys_atrs])

# print(train_df.columns.to_list())
# print(data_cls.__dict__)

start_time = print_msg('Постобработка датасетов с парами товаров')
expand_columns = ['main_pic', 'name_bert', 'epic']
train_df = data_cls.transform(None, train_df, expand_columns=expand_columns)
test_df = data_cls.transform(None, test_df, expand_columns=expand_columns)
train_df = data_cls.add_revers_pairs(train_df, test_df, number_fractions=41, all_rows=True)
print_time(start_time)

exclude_columns = data_cls.__dict__['exclude_columns']
exclude_columns.extend([cn for cn in data_cls.exclude_columns if cn not in exclude_columns])

data_cls.exclude_columns = exclude_columns

# с этой фичей лучше
# if 'cat32' in train_df.columns:
#     train_df.drop('cat32', axis=1, inplace=True)
#     test_df.drop('cat32', axis=1, inplace=True)
#     exclude_columns.append('cat32')
#     if 'cat32' in data_cls.category_columns:
#         data_cls.category_columns.remove('cat32')

cat_columns = data_cls.category_columns
model_columns = train_df.columns.to_list()

print('Обучаюсь на колонках:', model_columns)
print('Категорийные колонки:', cat_columns)
print('Исключенные колонки:', data_cls.exclude_columns)

if 'cat31' in train_df.columns:
    # добавление cat31 для валидации кто попался только один раз
    for cat31 in train_df.cat31.unique():
        if len(train_df.loc[train_df.cat31 == cat31]) < 5:
            print(f'cat31 = {cat31}')
            train_df = train_df.append(train_df.loc[train_df.cat31 == cat31])
            train_df = train_df.append(train_df.loc[train_df.cat31 == cat31])
            train_df = train_df.append(train_df.loc[train_df.cat31 == cat31])
        elif len(train_df.loc[train_df.cat31 == cat31]) < 10:
            print(f'cat31 = {cat31}')
            train_df = train_df.append(train_df.loc[train_df.cat31 == cat31])

    train_df.reset_index(drop=True, inplace=True)

print(f'Размер train_df = {train_df.shape}, test_df = {test_df.shape}')

columns_no_to_model = ['target', 'variantid1', 'variantid2']
trn = train_df[columns_no_to_model + ['cat3_grouped', 'cat31']]
train = train_df.drop(columns_no_to_model, axis=1)
target = train_df['target']

# print(train[cat_columns].dtypes)
# print(test_df[cat_columns].dtypes)

# test_df.drop(['variantid1', 'variantid2'], axis=1, inplace=True)

test_sizes = (0.1, 0.14, 0.15, 0.15000000000000002)
test_sizes = (0.14,)
# test_sizes = np.linspace(0.2, 0.3, 11)
# for num_iters in range(50, 151, 50):
# for SEED in range(100):
# for num_leaves in range(20, 51, 5):
# for num_iters in range(200, 201, 50):
for test_size in test_sizes:

    max_num += 1

    # test_size = round(test_size, 2)
    test_size = 0.14
    # test_size = 0.1
    # test_size = 0.15000000000000002
    num_iters = 555
    num_iters = 800
    SEED = 17

    print(f'test_size: {test_size} SEED={SEED} num_iters={num_iters}')

    stratify_columns = ['target', 'cat3_grouped']
    # stratify_columns = ['target', 'cat31']

    # Split the train_df into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(train,
                                                          target,
                                                          test_size=test_size,
                                                          stratify=trn[stratify_columns],
                                                          random_state=SEED)

    pool_train = lg.Dataset(data=X_train, label=y_train, free_raw_data=False,
                            feature_name=model_columns,
                            categorical_feature=cat_columns)
    pool_valid = lg.Dataset(data=X_valid, label=y_valid, free_raw_data=False,
                            feature_name=model_columns,
                            categorical_feature=cat_columns)

    num_folds = 4
    skf = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)
    split_kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)

    fit_on_full_train = False
    use_grid_search = False
    use_cv_folds = False
    build_model = True
    stratified = True
    write_log = False

    models, models_scores, predict_scores = [], [], []

    clf_params = dict(objective="binary",
                      # class_weight='balanced',
                      # is_unbalance=True,
                      # learning_rate=0.01,
                      num_iterations=num_iters,
                      # num_leaves=num_leaves,
                      # num_leaves=63,
                      # max_depth=10,
                      # max_depth=5,
                      # seed=42,
                      seed=SEED,
                      early_stopping_round=50,
                      n_jobs=7,
                      # verbose=-1,
                      verbose=50,
                      # device="gpu",
                      # gpu_platform_id=0,
                      # gpu_device_id=0,
                      )

    clf = lg.LGBMClassifier(**clf_params)

    if use_grid_search:
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            direction="minimize",
        )
        study.optimize(objective, n_trials=40)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        best_params = trial.params

        clf_params.update(best_params)
        print('clf_params', clf_params)

        clf = lg.LGBMClassifier(**clf_params)

    if use_cv_folds:
        if stratified:
            skf_folds = skf.split(train, trn[['target', 'cat3_grouped']])
        else:
            skf_folds = split_kf.split(train)

        for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
            print(f'Фолд {idx} из {num_folds}')
            X_train = train.iloc[train_idx]
            X_valid = train.iloc[valid_idx]
            y_train = target.iloc[train_idx]
            y_valid = target.iloc[valid_idx]

            clf = lg.LGBMClassifier(**clf_params)

            clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                    eval_metric='binary_logloss',
                    )
            models.append(clf)
            if build_model:
                DTS = (X_train, X_valid, y_train, y_valid, train, target, test_df, trn)
                predict_scores = predict_test(idx, clf, DTS, max_num)
                models_scores.append(predict_scores)
                acc_train, acc_valid, roc_auc, f1w, score_train, score_valid = predict_scores
                comment = {'test_size': test_size,
                           'SEED': SEED,
                           'size': f'pool_{idx}'}
                comment.update(data_cls.comment)
                comment.update({'stratified': stratify_columns})
                comment.update(clf.get_params())

                with open(file_logs, mode='a') as log:
                    # log.write('num;mdl;roc_auc;acc_train;acc_valid;sc_train;score;WF1;'
                    #           'model_columns;exclude_columns;cat_columns;comment\n')
                    log.write(f'{max_num};lg;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                              f'{score_train:.6f};{score_valid:.6f};{f1w:.6f};'
                              f'{train_df.columns.tolist()};'
                              f'{data_cls.exclude_columns};{cat_columns};{comment}\n')

        best_params = {'iterations': [clf.best_iteration_ for clf in models]}

    else:
        DTS = (X_train, X_valid, y_train, y_valid, train, target, test_df, trn)

        clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                eval_metric='binary_logloss',
                )

        print(clf.best_iteration_)
        print(clf.get_params())

        best_params = {'clf_iters': clf.best_iteration_,
                       'clf_lr': clf.get_params()['learning_rate']}

        models.append(clf)

        if build_model:
            if not fit_on_full_train:
                predict_scores = predict_test(0, clf, DTS, max_num)

            else:
                predict_scores = predict_test('pool', clf, DTS, max_num)
                acc_train, acc_valid, roc_auc, f1w, score_train, score_valid = predict_scores
                comment = {'test_size': test_size,
                           'SEED': SEED,
                           'size': 'pool'}
                comment.update(data_cls.comment)
                comment.update({'stratified': stratify_columns})
                comment.update(models[0].get_params())

                with open(file_logs, mode='a') as log:
                    # log.write('num;mdl;roc_auc;acc_train;acc_valid;sc_train;score;WF1;'
                    #           'model_columns;exclude_columns;cat_columns;comment\n')
                    log.write(f'{max_num};lg;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                              f'{score_train:.6f};{score_valid:.6f};{f1w:.6f};'
                              f'{train_df.columns.tolist()};'
                              f'{data_cls.exclude_columns};{cat_columns};{comment}\n')

                # обучение на всем трейне
                print('Обучаюсь на всём трейне...')
                model = lg.LGBMClassifier(objective="binary",
                                          seed=SEED,
                                          early_stopping_round=50,
                                          # device_type='cuda',
                                          # device_type='gpu',
                                          )

                model.fit(train, target, verbose=50, cat_features=cat_columns)
                predict_scores = predict_test('full', model, DTS, max_num)

        elif write_log:
            predict_scores = predict_train_valid(0, clf, DTS, max_num)

        best_params.update(clf.get_params())

    print('best_params:', best_params)

    if build_model or write_log:
        if len(models) > 1:
            predict_scores = [np.mean(arg) for arg in zip(*models_scores)]

        acc_train, acc_valid, roc_auc, f1w, score_train, score_valid = predict_scores

        print(f'Weighted F1-score = {f1w:.6f}')
        print('Параметры модели:', clf.get_params())

        print_time(start_time)

        comment = {'test_size': test_size,
                   'SEED': SEED,
                   'clf_iters': models[0].best_iteration_,
                   'clf_lr': models[0].get_params()['learning_rate'],
                   'stratified': stratify_columns}
        comment.update(data_cls.comment)
        comment.update(models[0].get_params())

        with open(file_logs, mode='a') as log:
            # log.write('num;mdl;roc_auc;acc_train;acc_valid;sc_train;score;WF1;'
            #           'model_columns;exclude_columns;cat_columns;comment\n')
            log.write(f'{max_num};lg;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                      f'{score_train:.6f};{score_valid:.6f};{f1w:.6f};'
                      f'{train_df.columns.tolist()};'
                      f'{data_cls.exclude_columns};{cat_columns};{comment}\n')
