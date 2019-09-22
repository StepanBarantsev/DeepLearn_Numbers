import numpy as np
from skimage.io import imread, imsave
from helper import parse_img, parse_labels, write_weigth_to_file, get_weight


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
                self.weights.append(get_weight(filename, self.shapes[i][0], self.shapes[i][1]))
        else:
            for i in range(len(self.filenames)):
                self.weights.append(get_weight(None, self.shapes[i][0], self.shapes[i][1]))

    def write_weights_to_file(self):
        for i in range(len(self.filenames)):
            write_weigth_to_file(self.filenames[i], self.weights[i])

    def learn(self, iterations):
        # Получам пикчи и ожидаемые результаты
        array_imgs = parse_img('train-images-idx3-ubyte', 60000)
        expected = parse_labels('train-labels-idx1-ubyte')
        for something in range(iterations):
            for index, img in enumerate(array_imgs):
                exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                exp[expected[index]] += 1
                alpha = 0.0000000001
                layers = [np.array(img)]
                for i in range(len(self.weights)):
                    layers.append(np.array(layers[i].dot(self.weights[i])))
                layers = np.array(layers)
                final_res = layers[len(layers) - 1]
                deltas = [(np.array(final_res - exp))]
                # У нас должно в итоге быть 2 дельты. На 0 вес умножать не нужно. Поэтому -1
                for i in range(len(self.weights) - 1):
                    deltas.append(np.array(deltas[i].dot(self.weights[len(self.weights) - 1 - i].T)))
                deltas = np.array(deltas)
                for i in range(len(deltas) // 2):
                    deltas[i], deltas[len(deltas) - 1 - i] = deltas[len(deltas) - 1 - i], deltas[i]
                for i in range(len(self.weights)):
                    # Надо делать матрицы иначе не работает транспонирование!
                    deltas[i] = np.matrix(deltas[i])
                    layers[i] = np.matrix(layers[i]).T
                    self.weights[i] -= alpha * layers[i].dot(deltas[i])
                    print(self.weights)
            print('Итерация номер %s' % something)
            self.write_weights_to_file()


o = Neuro2(['w1.txt', 'w2.txt'], weights_from_file=False, shapes=[(28 * 28, 40), (40, 10)])
o.learn(1)
o.write_weights_to_file()




