import numpy as np
from skimage.io import imread, imsave
from helper import parse_img, parse_labels


class Neuro:

    # filename из инита это имя файла с весами!!!
    def __init__(self, filename, weights_from_file):
        self.filename = filename
        if weights_from_file:
            self.weights = self.get_weights(from_file=True)
        else:
            self.weights = self.get_weights(from_file=False)

    def write_weigths_to_file(self):
        with open(self.filename, 'w') as f:
            self.weights = list(self.weights)
            for i in range(len(self.weights)):
                self.weights[i] = list(self.weights[i])
                for k in range(len(self.weights[i])):
                    self.weights[i][k] = str(self.weights[i][k])
            print(self.weights)
            weights = [' '.join(weight) for weight in self.weights]
            weights = '\n'.join(weights)
            f.write(weights)

    def get_weights(self, from_file=False):
        if from_file:
            with open(self.filename, 'r') as f:
                text = f.read()
                text = text.split('\n')
                weights = [i.split() for i in text]

            for i in range(len(weights)):
                for k in range(len(weights[i])):
                    weights[i][k] = float(weights[i][k])

            return np.array(weights)
        else:
            return np.array([[0.0 for i in range(10)] for k in range(28 * 28)])

    def learn(self, iterations):
        # Получам пикчи и ожидаемые результаты
        array_imgs = parse_img('train-images-idx3-ubyte', 60000)
        expected = parse_labels('train-labels-idx1-ubyte')
        for something in range(iterations):
            for index, img in enumerate(array_imgs):
                # Результат это вектор 1 10 в котором написана вероятность появления каждой из цифр
                result = img.dot(self.weights)
                # Ожидание по нулям везде кроме верной цифры
                exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                exp[expected[index]] += 1
                #  Получаем 10 ошибок разностей
                alpha = 0.0000001
                errors = np.matrix((result - exp) * alpha)
                # Этот подход работает! Значит есть возможность обобщения на n, нужно только понять как
                img = np.matrix(img).T
                self.weights -= img.dot(errors)
            print('Итерация номер %s' % something)
            self.write_weigths_to_file()

    def my_learn(self, iterations):
        # Получам пикчи и ожидаемые результаты
        array_imgs = parse_img('train-images-idx3-ubyte', 60000)
        expected = parse_labels('train-labels-idx1-ubyte')
        for something in range(iterations):
            for index, img in enumerate(array_imgs):
                exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                exp[expected[index]] += 1
                for index_pixel, pixel in enumerate(img):
                    self.weights[index_pixel] += pixel * exp
            print('Итерация номер %s' % something)
            self.write_weigths_to_file()

    # Рисует и сохраняет несколько картинок. Это нужно чисто для отладки
    @staticmethod
    def create_imgs(filename, num):
        with open(filename, 'rb') as f:
            text = f.read()
            barr = bytearray(text)[16:]
            for z in range(num):
                array = []
                for i in range(28):
                    array.append([])
                    for k in range(28):
                        array[i].append(barr[784 * z + i * 28 + k])
                imsave('pic%s.png' % z, np.array(array))

    @staticmethod
    def revert_grey(filename):
        img = imread(filename, as_gray=True)
        imsave(filename, img)

    # Предсказывает именно по картинке. Предсказывает по бд минст другой метод
    def predict_by_img(self, pic_predict_name):
        self.revert_grey(pic_predict_name)
        img = imread(pic_predict_name)
        # Двумерное изображение в вектор
        array = []
        for i in img:
            for k in i:
                array.append(k)

        array = np.array(array)
        ind, m = self.predict(array)
        print('Число на картинке это %s. Вероятность этого %s' % (ind, m))

    # Принимает на вход массив нампай
    def predict(self, img):
        result = img.dot(self.weights)
        m = result[0]
        ind = 0
        for i, res in enumerate(result):
            if res > m:
                m = res
                ind = i
        return ind, m

    # Вихузуализирует полученные картинки из весов
    def visualise(self):
        imgs = [[], [], [], [], [], [], [], [], [], []]

        for w in self.weights:
            for n in range(len(w)):
                imgs[n].append(w[n])

        # Преобразуем из векторов в матрицы
        # Я понимаю что это какой то говнокод но какая разница то лол)
        # Прога будет работать на 20 сек дольше, кому какая разница
        imgs2 = []
        for z in range(len(imgs)):
            imgs2.append([])
            for i in range(28):
                imgs2[z].append([])
                for k in range(28):
                    imgs2[z][i].append(imgs[z][i * 28 + k])

        imgs2 = np.array(imgs2)

        for index, img in enumerate(imgs2):
            print(img)
            imsave('a%s.png' % index, img)

    # Предсказывает из мниста
    def predict_by_mnist(self):
        array_imgs = parse_img('t10k-images-idx3-ubyte', 10000)
        expected = parse_labels('t10k-labels-idx1-ubyte')
        error = 0
        for i, img in enumerate(array_imgs):
            ind, m = self.predict(img)
            print('Число на картинке это %s. На самом деле %s' % (ind, expected[i]))
            if ind != expected[i]:
                error += 1
        print('Количество ошибок %s' % error)
        print('Всего чисел было %s' % len(expected))

o = Neuro('f.txt', True)
o.predict_by_mnist()