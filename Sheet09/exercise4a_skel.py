from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib import colormaps

preprocess_input = tf.keras.applications.resnet50.preprocess_input
decode_predictions = tf.keras.applications.resnet50.decode_predictions

def get_model() -> tf.keras.Model:
    """
    Load the ResNet50 model of TensorFlow Keras.

    Returns
    -------
    tf.keras.Model
        TensorFlow ResNet50 model pre-trained on the imgenet dataset.
    """

    return tf.keras.applications.resnet50.ResNet50()

def get_image(img_path: str, size: tuple[int, int]) -> np.ndarray:
    """
    Load an image from a given path with a given size
    as TensorFlow Tensor.

    Parameters
    ----------
    img_path : str
        (Relative) path to the image.
    size : tuple[int, int]
        Size of the output. If the size does not match
        the image size, the image gets reshaped to this
        size.

    Returns
    -------
    np.ndarray
        Preprocessed image as 4-dimensional numpy array.
        The preprocessed array has entries in [0,1].
    """

    # Load image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # Cast img to a numpy array with shape (3, size[0], size[1])
    img = tf.keras.preprocessing.image.img_to_array(img)
    # Transform img to a 4-tensor of shape (1, 3, size[0], size[1])
    img = tf.expand_dims(img, axis=0)
    # Cast to float32, if not done yet
    img = tf.cast(img, dtype=tf.float32)

    return preprocess_input(img)

def get_cam_model(model: tf.keras.Model, last_conv_layer: tf.keras.layers.Layer) -> tf.keras.Model:
    """
    Generate the CAM model which additionaly 
    outputs the activation of last convolutional layer.

    Parameters
    ----------
    model : tf.keras.Model
        Base model containing a classification part 
        with a GAP layer and a SoftMax layer.
    last_conv_layer : tf.keras.layers.Layer
        Last convolutional layer

    Returns
    -------
    tf.keras.Model
        Keras model which computes the output of `model` and 
        the last convolutional layer.
    """

    getter = tf.keras.Model([model.layers[0].input],
                            [last_conv_layer.output,
                             model.layers[-1].output])

    return getter

def analyze_predictions(predictions: np.ndarray) -> int:
    """
    Analyze the prediction of an image net model. 

    Parameters
    ----------
    predictions : np.ndarray
        Prediction returned by a keras model trained on 
        the imagenet dataset.

    Returns
    -------
    int
        Predicted class.
    """

    predicted_class = predictions.argmax()
    top5 = decode_predictions(predictions, top=5)[0]
    print("=========== resulting top predictions: ===========")
    for i in range(5):
        print("{}: probability {:6.2f}% for the class {}".format(
            i + 1, 100 * top5[i][2], top5[i][1]))
    return predicted_class

def get_cam_heatmap(img: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str) -> np.ndarray:
    """
    Get the CAM Heatmap for a given model with last 
    last convolutional layer name last_conv_layer_name

    Parameters
    ----------
    img : np.ndarray
        Input image.
    model : tf.keras.Model
        Base model for the CAM Heatmap
    last_conv_layer_name : str
        Name of the last convolutional layer. 
        This can be obtained from the method 
        `model.summary`.

    Returns
    -------
    np.ndarray
        Numpy array containing the image data of the heatmap.
    """

    last_conv_layer = model.get_layer(last_conv_layer_name)
    prediction_layer = model.layers[-1]

    # Get the output of the last convolutional layer and the
    # predicted class
    cam_model = get_cam_model(model, last_conv_layer)
    last_conv_out, predictions = cam_model(img)
    predicted_class = analyze_predictions(predictions.numpy())

    # Get the weights of the prediction layer
    W, b = prediction_layer.get_weights()

    # Get the channelwise product of the weights for
    # the predicted class and the
    # output of the last convolutional layer
    column = W[:, [predicted_class]]
    heatmap = last_conv_out @ column

    return heatmap[0, :, :, 0].numpy()

def get_gradient_cam_heatmap(img: np.ndarray, model: tf.keras.Model,
                             last_conv_layer_name: str,
                             classifier_layer_names: list[str]) -> np.ndarray:
    """
    Get the GradCam heatmap for a given model.

    Create an activity heatmap based on the gradients 
    of the gradients of the last convolutional layer 
    outputs of a network w.r.t. the classification output. 

    Parameters
    ----------
    img : np.ndarray
        Input image.
    model : tf.keras.Model
        Base CNN model.
    last_conv_layer_name : str
        Name of the last convolutional layer. 
        This can be obtained from the method 
        `model.summary`.
    classifier_layer_names : list[str]
        Names of the layers of the classification
        part of `model`. This can be obtained from 
        the method `model.summary`.

    Returns
    -------
    np.ndarray
        Numpy array containing the image data of the heatmap.
    """

    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Splitting the network into convolutional and classifier part
    # Model for the convolutional part
    convolutional_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # Model for the classifier part
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    # Track gradients with GradientTape()
    with tf.GradientTape() as tape:

        # Call the convolutional part with img as input
        last_conv_out = convolutional_model(img)

        # Track the derivatives with respect to the output of the
        # last convolutional layer.
        tape.watch(last_conv_out)

        # Call the classifier part with last_conv_out as input
        predictions = classifier_model(last_conv_out)

        # Analyze the predictions, and get winning class data
        predicted_class = analyze_predictions(predictions.numpy())
        top_class_channel = predictions[:, predicted_class]

    # Get the derivatives of the predicted class channel w.r.t. the output
    # of the last convolutional layer
    gradient = tape.gradient(top_class_channel, last_conv_out)

    # gradient is of shape (1, oh, ow, c) where oh, ow are the height and
    # width of the outout of the last convolutional layer. The average over
    # the first three axes has to be taken.
    pooled_gradient = tf.reduce_mean(gradient, axis=(0, 1, 2))

    # Compute heatmap
    heatmap = last_conv_out * pooled_gradient
    heatmap = tf.reduce_sum(heatmap, axis=(0, 3))

    return heatmap.numpy()

def superimpose_heatmap(img_path: str, heatmap: np.ndarray, alpha: float = .7) -> np.ndarray:
    """
    Superimpose an image with a heatmap. 

    Parameters
    ----------
    img_path : str
        Path to the image.
    heatmap : np.ndarray
        Heatmap as numpy array. 
    alpha : float
        Transparency parameter, by default .7.
        If set to 0, the heatmap can not be seen,
        if set to 1, the image is fully superimposed 
        by the heatmap.

    Returns
    -------
    np.ndarray
        Numpy array containing the data of the superimposition.
    """

    # load image, e.g., float array of shape (465, 621, 3)
    image_np = plt.imread(img_path).astype(np.float32)
    heatmap_uint8 = np.uint8(np.maximum(heatmap, 0) / heatmap.max() * 255)
    cm_jet = colormaps.get_cmap("jet")
    jet_colors = cm_jet(np.arange(256))[:, :3]
    heatmap_jet = jet_colors[heatmap_uint8]

    # scale color heatmap to shape (465, 621, 3)
    target_h, target_w, _ = image_np.shape
    h, w, _ = heatmap_jet.shape
    heatmap_scaled = zoom(heatmap_jet, (target_h/h, target_w/w, 1))
    heatmap_scaled_uint8 = np.uint8(np.maximum(
        heatmap_scaled, 0) / heatmap_scaled.max() * 255)

    # superimpose image and heatmap
    return np.uint8(image_np * (1 - alpha) + heatmap_scaled_uint8 * alpha)

def show_image(image_path: str, title: Optional[str] = None) -> None:
    """
    Use matplotlibs imshow routine to display an image.

    Parameters
    ----------
    image_path : str
        Path to the image
    title : str, optional
        Title of the image displayed above.
    """
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.title(title)
    plt.show()

def show_heatmap(heatmap: np.ndarray, title: Optional[str] = None) -> None:
    """
    Use matplotlibs imshow routine to display a heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        Numpy array containing the heatmap data.
    title : Optional[str], optional
        Title displayed above the shown image.
    """

    plt.imshow(heatmap, cmap='jet')
    plt.title(title)
    plt.show()

if __name__ == '__main__':

    # Choose an input image
    img_path = 'truck.jpg'
    

    model = tf.keras.applications.resnet50.ResNet50(include_top=True)
    
    # TODO Use model.summary() to obtain the input size of the 
    # network, the name of the last convolutional layer, and the 
    # name of the classification layers, i.e. the name of the GAP 
    # layer and the SoftMax layer.
    resnet_size = None 
    last_conv_layer_name = None
    classifier_layer_names = None

    img = get_image(img_path, resnet_size)

    heatmap_cam = get_cam_heatmap(img, model, last_conv_layer_name)
    heatmap_grad_cam = get_gradient_cam_heatmap(img, model,
                                                last_conv_layer_name,
                                                classifier_layer_names)

    image_cam = superimpose_heatmap(img_path, heatmap_cam)
    image_grad_cam = superimpose_heatmap(img_path, heatmap_grad_cam)

    plt.axis('off')
    plt.imshow(image_cam)
    plt.savefig('truck_cam.pdf')
    plt.close()

    plt.axis('off')
    plt.imshow(image_grad_cam)
    plt.savefig('truck_grad_cam.pdf')
    plt.close()
