from some_functions import read_train_test

__import__("warnings").filterwarnings('ignore')

if __name__ == '__main__':
    _, _, dts_obj = read_train_test(rebuilding_pairs=True)
    print(type(dts_obj))
