import os
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from scipy.misc import imresize  # version 1.1.0.
np.seterr(divide='ignore', invalid='ignore')


def make_tensor(image, out_dir=None, show=False, save=True):
    """
    Function to turn an image array into a tensor, suitable as input into keras model.

    Inputs:
        image: an image array of size (rows, columns, 3).
        out_dir: path to a local directory in which to save a png of the input image.
        show: if true, plots the image.
        save: if true, saves the image to disk.

    Returns:
        img_tensor: an array of size (1, rows, columns, 3)
    """

    # add a dimension because the model expects input shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(image, axis=0)

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    if save:
        plt.figure(figsize=(5, 6))
        plt.imshow(img_tensor[0])
        plt.axis('off')
        figs = plt.gcf()
        figs.savefig(os.path.join(out_dir, 'tested_image.png'))

    return img_tensor


def data_loader(name, normalize=True, categorical=True, NB_CLASSES=10, resize=False, res_shape=(128, 128)):
    """
    Function for loading the input dataset.

    Inputs:
            name: name of the dataset
            normalize: boolean variable for normalsing the dataset
            categorical: boolean variable for converting the lables to categorical
            NB_CLASSES: number of classes in the input dataset
            resize: boolean variable for resizing the images of the input dataset to any size
            res_shape: a tuple with the desired resizing shape

    Outputs: a tuple with the training and testing images and lables
    """

    if name == 'CIFAR10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        print(X_train.shape, 'training samples')
        print(X_test.shape, 'test samples')

    else:
        # you can add your own dataset here
        raise ValueError('Not supported dataset')

    # convert to categorical
    if categorical:
        Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
        Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

    # Resize training images
    if resize:
        img_rows = res_shape[0]
        img_cols = res_shape[1]
        X_train = np.array([imresize(img, size=(img_rows, img_cols)) for img in X_train])
        X_test = np.array([imresize(img, size=(img_rows, img_cols)) for img in X_test])

    # Normalization step
    if normalize:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':

    dest_dir = os.path.join('visualisations_activations')

    # load the data
    x_train, y_train, x_test, y_test = data_loader('CIFAR10', resize=False)

    # load a model
    model = load_model('CNN_80epochs.h5')
    # model.summary()

    # choose a random image from the test set
    im = x_test[50]
    img_tensor = make_tensor(im, out_dir=dest_dir, save=True)

    # extract the outputs of the top ten layers
    layer_outputs = [layer.output for layer in model.layers[:10]]

    # creates a model that will return these outputs, given the model input
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # returns a list of Numpy arrays: one array per layer activation
    activations = activation_model.predict(img_tensor)

    # visualizing every channel in every intermediate activation

    # names of the layers made accessible for plotting
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    # displays the feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]  # number of features in the feature map
        size = layer_activation.shape[1]  # the feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # tile the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):  # tile each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()  # post-process the feature to make it nicer visually
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                # display the grid
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        # create the plot
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        fig = plt.gcf()
        fig.savefig(os.path.join(dest_dir, layer_name + '.png'))

    print('Finished successfully')
