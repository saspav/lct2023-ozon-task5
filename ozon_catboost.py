import re
import numpy as np
import pandas as pd

from pathlib import Path

import optuna
from optuna.integration import CatBoostPruningCallback

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from data_process import DataTransform

from some_functions import read_train_test, read_train_test_sample
from some_functions import MODELS_LOGS, predict_test, predict_train_valid, get_max_num
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')


def objective(trial: optuna.Trial) -> float:
    param = {
        "loss_function": trial.suggest_categorical("loss_function",
                                                   ["binary"]),
        "auto_class_weights": trial.suggest_categorical("auto_class_weights",
                                                        [None, "Balanced"]),
        # "iterations": trial.suggest_int("iterations", 200, 2000, step=200),
        # "depth": trial.suggest_int("depth", 1, 12),
        # "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.2, step=0.05),
    }

    gbm = CatBoostClassifier(cat_features=cat_columns,
                             eval_metric='TotalF1',
                             early_stopping_rounds=80,
                             random_seed=42,
                             # task_type="GPU",
                             # devices='0:1',
                             **param)

    pruning_callback = CatBoostPruningCallback(trial, "TotalF1")

    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
        early_stopping_rounds=80,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    accuracy = accuracy_score(y_valid, gbm.predict(X_valid))
    return accuracy


file_logs = MODELS_LOGS
max_num = get_max_num(file_logs)

start_time = print_msg('Обучение Catboost классификатор...')

data_cls = DataTransform()

# # небольшая выборка для тестов
# train_df, test_df, dts = read_train_test_sample(rebuilding_pairs=True)

# чтение подготовленного датасета
train_df, test_df, dts = read_train_test(rebuilding_pairs=False)

cam_atrs = sorted(filter(lambda x: re.fullmatch(r'atr_\d+', x), train_df.columns))
fuz_atts = sorted(filter(lambda x: re.fullmatch(r'fuz_atr_\d+', x), train_df.columns))
comp_atrs = sorted(filter(lambda x: re.fullmatch(r'comp_atr_\d+', x), train_df.columns))
keys_atrs = sorted(filter(lambda x: re.fullmatch(r'keys_\d+', x), train_df.columns))

# количество характеристик товара, которые будем использовать: задавать четное число
# было 176
numbers_cam_atrs = 276  # 99 * 2
numbers_fuz_atrs = 276  # 99 * 2
numbers_comp_atrs = 276  # 99 * 2
numbers_keys_atrs = 276  # 99 * 2
# перезапишем словарь из класса dts в класс data_cls
dts.exclude_columns.extend(['pic_emb_mean', 'cam_lemm', 'name_stem_lemm',
                            # 'cam_lm_len',  # эта колонка была в сабмите 207
                            'cos_pics', 'epic'])
dts.exclude_columns.extend(cam_atrs[numbers_cam_atrs:])
dts.exclude_columns.extend(fuz_atts[numbers_fuz_atrs:])
dts.exclude_columns.extend(comp_atrs[numbers_comp_atrs:])
dts.exclude_columns.extend(keys_atrs[numbers_keys_atrs:])

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
print('category_columns:', data_cls.__dict__['category_columns'])

start_time = print_msg('Постобработка датасетов с парами товаров')
expand_columns = ['main_pic', 'name_bert']
train_df = data_cls.transform(None, train_df, expand_columns=expand_columns)
test_df = data_cls.transform(None, test_df, expand_columns=expand_columns)
train_df = data_cls.add_revers_pairs(train_df, test_df, number_fractions=41, all_rows=True)
print_time(start_time)

exclude_columns = data_cls.__dict__['exclude_columns']
exclude_columns.extend([cn for cn in data_cls.exclude_columns if cn not in exclude_columns])

data_cls.exclude_columns = exclude_columns

if 'cat32' in train_df.columns:
    train_df.drop('cat32', axis=1, inplace=True)
    test_df.drop('cat32', axis=1, inplace=True)
    exclude_columns.append('cat32')
    if 'cat32' in data_cls.category_columns:
        data_cls.category_columns.remove('cat32')

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

# test_sizes = (0.1, 0.14, 0.15)
test_sizes = (0.1,)
# test_sizes = np.linspace(0.1, 0.4, 4)
# for num_iters in range(50, 151, 50):
# for SEED in range(100):
for test_size in test_sizes:
    # for num_iters in range(200, 901, 50):
    # for num_leaves in range(20, 51, 5):
    max_num += 1

    # test_size = 0.2
    # num_iters = 600
    SEED = 86

    print(f'test_size: {test_size} SEED={SEED}')

    stratify_columns = ['target', 'cat3_grouped']
    # stratify_columns = ['target', 'cat31']

    # Split the train_df into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(train,
                                                          target,
                                                          test_size=test_size,
                                                          stratify=trn[stratify_columns],
                                                          random_state=SEED)

    pool_train = Pool(data=X_train, label=y_train, cat_features=cat_columns)
    pool_valid = Pool(data=X_valid, label=y_valid, cat_features=cat_columns)
    pool_test = Pool(data=test_df.drop(columns_no_to_model[1:], axis=1),
                     cat_features=cat_columns)

    num_folds = 4
    skf = StratifiedKFold(n_splits=num_folds)
    split_kf = KFold(n_splits=num_folds)

    fit_on_full_train = False
    use_grid_search = False
    use_cv_folds = False
    build_model = True
    stratified = True
    write_log = False

    models, models_scores, predict_scores = [], [], []

    loss_function = 'Logloss'

    # auto_class_weights = 'Balanced'
    auto_class_weights = None

    eval_metric = 'Precision'
    # eval_metric = 'TotalF1'

    clf_params = dict(cat_features=cat_columns,
                      auto_class_weights=auto_class_weights,
                      loss_function=loss_function,
                      eval_metric=eval_metric,
                      # iterations=2000,  # попробовать столько итераций
                      early_stopping_rounds=80,
                      random_seed=SEED,
                      # task_type="GPU",
                      # devices='0:1',
                      )

    clf = CatBoostClassifier(**clf_params)

    if use_grid_search:
        # grid_params = {
        #     'max_depth': [5, 6],
        #     'learning_rate': [0.1, 0.15, 0.2],
        # }
        # grid_search_result = clf.grid_search(grid_params, train, target,
        #                                      cv=skf,
        #                                      stratified=True,
        #                                      refit=True,
        #                                      plot=True,
        #                                      verbose=100,
        #                                      )
        # best_params = grid_search_result['params']
        # models.append(clf)

        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
        )
        study.optimize(objective, n_trials=12, timeout=600)

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

        clf = CatBoostClassifier(**clf_params)

    if use_cv_folds:
        if stratified:
            skf_folds = skf.split(train, trn[['target', 'cat3_grouped']])
        else:
            skf_folds = split_kf.split(train)

        for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
            print(f'Фолд {idx} из {num_folds}')
            train_data = Pool(data=train.iloc[train_idx],
                              label=target.iloc[train_idx],
                              cat_features=cat_columns)
            valid_data = Pool(data=train.iloc[valid_idx],
                              label=target.iloc[valid_idx],
                              cat_features=cat_columns)
            model = clf
            model.fit(train_data, eval_set=valid_data, use_best_model=True, verbose=100)
            models.append(model)
            if build_model:
                DTS = (X_train, X_valid, y_train, y_valid, train, target, test_df, trn)
                predict_scores = predict_test(idx, clf, DTS, max_num, submit_prefix='cb_')
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
                    log.write(f'{max_num};cb;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                              f'{score_train:.6f};{score_valid:.6f};{f1w:.6f};'
                              f'{train_df.columns.tolist()};'
                              f'{data_cls.exclude_columns};{cat_columns};{comment}\n')

        best_params = {'iterations': [clf.tree_count_ for clf in models]}

    else:
        DTS = (X_train, X_valid, y_train, y_valid, train, target, test_df, trn)

        clf.fit(pool_train, eval_set=pool_valid, use_best_model=True, verbose=50)

        models.append(clf)

        best_params = {'clf_iters': clf.tree_count_,
                       'clf_lr': clf.get_all_params()['learning_rate']}

        if build_model:
            if not fit_on_full_train:
                predict_scores = predict_test(0, clf, DTS, max_num, submit_prefix='cb_')

            else:
                predict_scores = predict_test('pool', clf, DTS, max_num, submit_prefix='cb_')
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
                clf_params['iterations'] = int(clf.tree_count_ * 1.1)
                clf_params['learning_rate'] = clf.get_all_params()['learning_rate']
                model = CatBoostClassifier(**clf_params)
                model.fit(train, target, verbose=50, cat_features=cat_columns)
                predict_scores = predict_test('full', model, DTS, max_num,
                                              submit_prefix='cb_')

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
                   'clf_iters': clf.best_iteration_,
                   'clf_lr': clf.get_params().get('learning_rate'),
                   'stratified': stratified}
        comment.update(data_cls.comment)
        comment.update(clf.get_params())

        with open(file_logs, mode='a') as log:
            # log.write('num;mdl;roc_auc;acc_train;acc_valid;sc_train;score;WF1;'
            #           'model_columns;exclude_columns;cat_columns;comment\n')
            log.write(f'{max_num};cb;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                      f'{score_train:.6f};{score_valid:.6f};{f1w:.6f};'
                      f'{train_df.columns.tolist()};'
                      f'{data_cls.exclude_columns};{cat_columns};{comment}\n')
