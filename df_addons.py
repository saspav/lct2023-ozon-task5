import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from time import time
from bisect import bisect_left
from print_time import print_time, print_msg
from mts_paths import WORK_PATH

__import__("warnings").filterwarnings('ignore')



def df_to_excel(save_df, file_excel, ins_col_width=None, float_cells=None):
    """ экспорт в эксель """
    writer = pd.ExcelWriter(file_excel, engine='xlsxwriter')
    # save_df.to_excel(file_writer, sheet_name='find_RFC', index=False)
    # Convert the dataframe to an XlsxWriter Excel object.
    # Note that we turn off the default header and skip one row to allow us
    # to insert a user defined header.
    save_df.to_excel(writer, sheet_name='logs', startrow=1, header=False, index=False)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['logs']
    # Add a header format.
    header_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 13,
        'bold': True,
        'text_wrap': True,
        'align': 'center',
        'valign': 'center',
        'border': 1})
    # Write the column headers with the defined format.
    worksheet.freeze_panes(1, 0)
    for col_num, value in enumerate(save_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    # вставка ссылок
    url_format = workbook.add_format({
        'font_color': 'blue',
        'underline': 1,
        'font_name': 'Times New Roman',
        'font_size': 13
    })
    cell_format = workbook.add_format()
    cell_format.set_font_name('Times New Roman')
    cell_format.set_font_size(13)
    float_format = workbook.add_format({'num_format': '0.000000'})
    float_format.set_font_name('Times New Roman')
    float_format.set_font_size(13)
    col_width = [7, 7, 10, 32, 8, 12, 12] + [32] * 9
    if ins_col_width:
        for num_pos, width in ins_col_width:
            col_width.insert(num_pos, width)
    for num, width in enumerate(col_width):
        now_cell_format = cell_format
        if float_cells and num in float_cells:
            now_cell_format = float_format
        worksheet.set_column(num, num, width, now_cell_format)
    worksheet.autofilter(0, 0, len(save_df) - 1, len(col_width) - 1)
    writer.close()


def memory_compression(df, use_category=True, use_float=True):
    """
    Изменение типов данных для экономии памяти
    :param df: исходный ДФ
    :param use_category: преобразовывать строки в категорию
    :param use_float: преобразовывать float в пониженную размерность
    :return: сжатый ДФ
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        # print(f'{col} тип: {tmp[col].dtype}', str(tmp[col].dtype)[:4])

        if str(df[col].dtype)[:4] in 'datetime':
            continue

        elif str(df[col].dtype) not in ('object', 'category'):
            col_min = df[col].min()
            col_max = df[col].max()
            if str(df[col].dtype)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and \
                        col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and \
                        col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and \
                        col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and \
                        col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif use_float and str(df[col].dtype)[:5] == 'float':
                if col_min > np.finfo(np.float16).min and \
                        col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and \
                        col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif use_category and str(df[col].dtype) == 'object':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Исходный размер датасета в памяти '
          f'равен {round(start_mem, 2)} мб.')
    print(f'Конечный размер датасета в памяти '
          f'равен {round(end_mem, 2)} мб.')
    print(f'Экономия памяти = {(1 - end_mem / start_mem):.1%}')
    return df
