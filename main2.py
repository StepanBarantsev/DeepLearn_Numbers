import numpy as np
from skimage.io import imread, imsave
from helper import parse_img, parse_labels


class Neuro2:

    # Количество слоев равно количеству имен файлов
    # weights_from_file -- булева переменная, отвечающая за то берем ли мы веса из файла
    # shapes -- кортежи содержащие в себе размерности матриц весов (список кортежей)
    def __init__(self, filenames, weights_from_file, shapes):
        self.filenames = filenames
        self.shapes = shapes
        self.weights = []
        self.get_weights(weights_from_file)

    def get_weights(self, from_file):
        if from_file:
            for i, filename in enumerate(self.filenames):
                self.weights.append(self.get_weight(filename, self.shapes[i][0], self.shapes[i][1]))
        else:
            for i in range(len(self.filenames)):
                self.weights.append(self.get_weight(None, self.shapes[i][0], self.shapes[i][1]))

    def write_weights_to_file(self):
        for i in range(len(self.filenames)):
            self.write_weigth_to_file(self.filenames[i], self.weights[i])

    @staticmethod
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

    @staticmethod
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
            return np.random.random((r1, r2))


o = Neuro2(['w1.txt', 'w2.txt'], weights_from_file=False, shapes=[(28 * 28, 40), (40, 10)])
o.write_weights_to_file()




