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
