import pandas as pd
from pathlib import Path
from glob import glob

from data_process import WORK_PATH
from print_time import print_msg

__import__("warnings").filterwarnings('ignore')

PREDICTIONS_DIR = WORK_PATH.joinpath('predictions')


def merge_files(name_files):
    df = pd.DataFrame()
    columns_to_submit = ['variantid1', 'variantid2', 'target']
    for filename in name_files:
        print('Обрабатываю файл:', Path(filename).name)
        temp = pd.read_csv(filename)
        if len(df):
            df = df.merge(temp[columns_to_submit], on=columns_to_submit[:2], how='left')
        else:
            df = temp[columns_to_submit]

    print('df.shape:', df.shape)

    target_cols = [*filter(lambda x: 'target' in x, df.columns)]
    print(target_cols)

    df['max_all'] = df[target_cols].max(axis=1)
    df['min_max'] = df[target_cols].apply(lambda x: max(x) if max(x) > 0.5 else min(x),
                                          axis=1)
    print(df)
    df.to_excel('mux.xlsx', index=False)

    max_num = 3
    for submit_prefix in ('max_all', 'min_max'):
        submit = df.copy()
        submit['target'] = submit[submit_prefix]
        submit_csv = f'{submit_prefix}_submit_{max_num}.csv'
        # Сохранение предсказаний в файл
        file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
        submit = submit[columns_to_submit]
        submit.to_csv(file_submit_csv, index=False)
    return df


start_time = print_msg('Поиск максимума среди классификаторов...')

stacking_dir = WORK_PATH.joinpath('best')
files_test = glob(f'{stacking_dir}/*submit*.csv')
df = merge_files(files_test)
