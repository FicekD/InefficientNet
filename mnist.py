import numpy as np
import os
import inefficientnet as ien
import matplotlib.pyplot as plt


def prepare_data(path, labels, img_size, channels=1):
    data = np.loadtxt(path, delimiter=',')
    # TODO: multidimensional data
    x = data[:, 1:].reshape((-1, ) + img_size) / 255
    y_raw = data[:, :1].astype(np.int32).reshape(-1)
    y = np.zeros((y_raw.size, labels))
    y[np.arange(y_raw.size), y_raw] = 1
    return x, y


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(base_path, 'data', 'mnist', 'mnist_train.csv')
    val_path = os.path.join(base_path, 'data', 'mnist', 'mnist_val.csv')
    test_path = os.path.join(base_path, 'data', 'mnist', 'mnist_test.csv')

    img_size = (28, 28)
    labels = 10

    train_x, train_y = prepare_data(train_path, labels, img_size)
    # val_x, val_y = prepare_data(val_path, labels, img_size)
    # test_x, test_y = prepare_data(test_path, labels, img_size)

    classifier = ien.SequentialModel()
    classifier.add([
        ien.Input(img_size),
        ien.Flatten(),
        ien.Dense(256, ien.LeakyReLU(0.01)),
        ien.Dense(128, ien.LeakyReLU(0.01)),
        ien.Dense(64, ien.LeakyReLU(0.01)),
        ien.Dense(labels, ien.Sigmoid)
    ])

    classifier.compile(ien.categorical_crossentropy, ien.GradientDescent(0.01))
    classifier.summary()
    classifier.fit(train_x, train_y, 50, 64)
    # classifier.evaluate(test_x, test_y, 64)

    classifier.plot_loss()

    idx = [1, 0, 16, 7, 3, 8, 21, 6, 10, 11]
    y_pred = classifier.predict(train_x[idx])
    plt.figure(2)
    for i, j in zip(range(10), idx):
        plt.subplot(2, 10, i+1)
        plt.imshow(train_x[j])
        plt.subplot(2, 10, 11+i)
        plt.bar(range(10), y_pred[i])
        plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    main()