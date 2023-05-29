from time import time


def convert_seconds(time_apply):
    # print(type(time_apply), time_apply)
    try:
        time_apply = float(time_apply)
    except ValueError:
        time_apply = 0
    if isinstance(time_apply, (int, float)):
        hrs = time_apply // 3600
        mns = time_apply % 3600
        sec = mns % 60
        time_string = ''
        if hrs:
            time_string = f'{hrs:.0f} час '
        if mns // 60 or hrs:
            time_string += f'{mns // 60:.0f} мин '
        return f'{time_string}{sec:.1f} сек'


def print_time(time_start):
    """
    Печать времени выполнения процесса
    :param time_start: время запуска в формате time.time()
    :return:
    """
    time_apply = time() - time_start
    print(f'Время обработки: {convert_seconds(time_apply)}')


def print_msg(msg):
    print(msg)
    return time()


if __name__ == "__main__":
    print(convert_seconds('165.31278347969055'))
