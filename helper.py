import numpy as np


def parse_labels(filename):
    with open(filename, 'rb') as f:
        text = f.read()
        barr = bytearray(text)
        array = []
        for byte in barr[8:]:
            array.append(byte)
        return np.array(array)


# Возвращает матрицу картинок
# Картинка это одномерный список, с которым при этом работать нужно как с двумерным
def parse_img(filename, num_of_imgs):
    with open(filename, 'rb') as f:
        text = f.read()
        barr = bytearray(text)[16:]
        array = []
        for z in range(num_of_imgs):
            array.append([])
            for i in range(28):
                for k in range(28):
                    array[z].append(barr[784 * z + i * 28 + k])
        return np.array(array)


def write_weigth_to_file(filename, weight):
    with open(filename, 'w') as f:
        weight = list(weight)
        for i in range(len(weight)):
            weight[i] = list(weight[i])
            for k in range(len(weight[i])):
                weight[i][k] = str(weight[i][k])
        print(weight)
        weight = [' '.join(w) for w in weight]
        weight = '\n'.join(weight)
        f.write(weight)


def get_weight(filename, r1, r2):
    if filename is not None:
        with open(filename, 'r') as f:
            text = f.read()
            text = text.split('\n')
            weights = [i.split() for i in text]

        for i in range(len(weights)):
            for k in range(len(weights[i])):
                weights[i][k] = float(weights[i][k])

        return np.array(weights)
    else:
        # Это плохо, но нам нужны детерменированные веса!
        return np.array([[0.0 for i in range(r2)] for k in range(r1)])
        # Тут надо настроить чтобы рандомились небольшие числа. Иначе корректировки просто их не могут победить
        # return np.random.random((r1, r2))
