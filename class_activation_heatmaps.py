import os
import cv2
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize  # version 1.1.0.
np.seterr(divide='ignore', invalid='ignore')


def plot_results(image, act_heatmap, prediction_text, layer_name, destination_dir):
    """
    Function to take an image, it's class activation map and class predictions and produces a visualisation showing the
    image, the heat-map and a composite image of the heat-map and image.

    Inputs:
        image: an image array of size (rows, cols, 3).
        act_heatmap: class activation heatmap - output of 'create_heatmap' function.
        prediction_text: sorted list containing (class, probability) tuples of 3 highest scoring classes.
        layer_name: string of keras layer name.
        destination_dir: path to a local directory to save a visualisation of the image and corresponding heatmap.

    Returns:
        None
    """

    # superimpose the heatmap on the image
    superimposed_img = act_heatmap * 0.4 + image

    # create the plot
    fig, ax = plt.subplots(1, 3)
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0].imshow(im)
    ax[0].set_title('Original')
    ax[0].text(0, (image.shape[1] + 3), str(prediction_text))
    ax[1].imshow(heatmap)
    ax[1].set_title('Activation heatmap')
    ax[2].imshow(superimposed_img)
    ax[2].set_title('Combined')

    # save the plot
    fig = plt.gcf()
    fig.savefig(os.path.join(destination_dir, layer_name + '.png'))
    plt.close(fig)


def create_heatmap(model, predictions, image_size=(32, 32), layer='conv2d_1'):
    """
    Function to create a class activation heat-map for a specfic keras model layer.

    Inputs:
        model: keras model.
        predictions: numpy array of predictions as output by the keras model.predict method.
        image_size: (rows, cols) of the input image.
        layer: string of keras layer name.

    Returns:
            rgb_heatmap: RGB image array of class activations
    """

    # the chosen class
    class_index = np.argmax(predictions[0])
    class_output = model.output[:, class_index]

    # output feature map of the chosen layer
    conv_layer = model.get_layer(layer)

    # gradient of the chosen class with regard to the output feature map of chosen layer
    grads = K.gradients(class_output, conv_layer.output)[0]

    # vector where each entry is the mean intensity of the gradient
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # the number of channels in the layer
    out_shape = int(pooled_grads.shape[0])

    # get quantities given a sample image
    iterate = K.function([model.input], [pooled_grads, conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

    # multiplies each channel in the feature map array to weight by importance to class
    for i in range(out_shape):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # mean of the resulting feature-map is the heat-map of class activation
    act_heatmap = np.mean(conv_layer_output_value, axis=-1)

    # normalise the heat-map
    act_heatmap = np.maximum(act_heatmap, 0)
    act_heatmap /= np.max(act_heatmap)

    # resize the heat-map to match the image
    act_heatmap = cv2.resize(act_heatmap, image_size)
    act_heatmap = cv2.normalize(act_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # convert to RGB
    cmap = plt.get_cmap('jet')
    rgba_heatmap = cmap(act_heatmap)
    rgb_heatmap = np.delete(rgba_heatmap, 3, 2)

    return rgb_heatmap


def decode_predictions(predictions):
    """
    Function to associate outputted predictions with target classes.

    Inputs:
        predictions: numpy array of predictions as output by the keras model.predict method.

    Outputs:
        preds_text: sorted list containing (class, probability) tuples of 3 highest scoring classes.
    """

    # CIFAR10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # associate predictions with correct class
    prediction_list = zip(classes, predictions[0])

    # sort in descending order (most likely prediction first)
    sorted_prediction_list = sorted(list(prediction_list), key=lambda x: x[1], reverse=True)

    # get top 3 results
    preds_text = []

    for i in range(3):
        preds_text.append(sorted_prediction_list[i])

    return preds_text


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
        plt.close(figs)

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

    Outputs: a tuple with the training and testing images and labels
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

    dst_dir = os.path.join('visualisations_heatmaps')

    # load the data
    x_train, y_train, x_test, y_test = data_loader('CIFAR10', resize=False, res_shape=(128, 128))

    # load a model and choose a layer to visualise
    # my_model = load_model('CNN_baseline_80epochs.h5')
    my_model = load_model('CNN_80epochs.h5')
    # my_model = load_model('mobilenet_80epochs.h5')

    # summarise the model (here you can find the layer names)
    my_model.summary()

    # choose a layer for which to visualise the activation heatmap
    chosen_layer = 'conv2d_17'

    # choose an image from the test set and transform to tensor for prediction
    im = x_test[903]
    img_tensor = make_tensor(im, out_dir=dst_dir, save=True)

    # ensure the image size matches that expected by the model
    input_shape = my_model.input_shape[1:3]
    img_shape = img_tensor.shape[1:3]

    if input_shape != img_shape:
        raise Exception('Image size is incorrect. The expected image size is {}. Please resize images using the data '
                        'loader function.'.format(input_shape))

    # make prediction for the image
    preds = my_model.predict(img_tensor)

    # get a user-friendly list of the top 3 predictions
    pred_text = decode_predictions(preds)

    # create the activation heatmap for a given layer
    heatmap = create_heatmap(my_model, preds, image_size=img_shape, layer=chosen_layer)

    # plot and save the results
    plot_results(im, heatmap, pred_text, chosen_layer, dst_dir)

    print('Finished successfully')





