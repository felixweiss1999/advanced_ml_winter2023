import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import decode_predictions

# import model to be attacked, pixels expected in range [-1, 1]
def load_models():
    model = EfficientNetV2B0(
        include_preprocessing=False)
    model_logits = EfficientNetV2B0(
        include_preprocessing=False,
        classifier_activation=None)
    return model, model_logits

# load image used for the attack
def load_image(name=None):
    if name is None:
        img = tf.Variable(np.random.randn(1, 224, 224, 3) / 10)
        img = tf.clip_by_value(img, -1, 1)
    else:
        img = image.load_img(name, target_size=(224, 224))
        img = image.img_to_array(img) * 2 / 255.0 - 1
        img = tf.expand_dims(img, axis=0)
    return img

def classify_image(img, model, save=False, name='file.pdf'):
    preds = model.predict(img)
    top5 = decode_predictions(preds, top=5)[0]
    print(top5)
    plt.imshow((img[0] + 1) / 2)
    plt.title(
        f'class = {top5[0][1]}, probability = {top5[0][2]:.2f}')
    if save:
        plt.savefig(name, bbox_inches='tight')

# define gradient
@tf.function
def img_gradient(x, y, model_logits, loss_fn):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = -loss_fn(y, model_logits(x))
    return tape.gradient(loss, x)

# perform attack, here via gradient ascent
def perform_attack(x, y, model_logits, loss_fn, eta=.5, steps=20):
    for k in range(steps):
        g = img_gradient(x, y, model_logits, loss_fn)
        x = x + g * eta
        x = tf.clip_by_value(x, -1, 1)
        classify_image(x, model)
        plt.pause(.1)
    return x

if __name__ == '__main__':

    # load models
    model, model_logits = load_models()

    # define loss function
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    # load image
    name = 'jpmzemke.jpg'
    #name = None
    img = load_image(name)

    # classify image using model, show result
    #classify_image(img, model, save=True, name='random_pre.pdf')
    #classify_image(img, model, save=True, name='jpmzemke_pre.pdf')
    classify_image(img, model)
    plt.show()

    # select class that the network should return
    y = 999 # toilet tissue

    # perform attack
    x = perform_attack(img, y, model_logits, loss_fn, eta=.5, steps=20)

    # classify attacked image using model, show result
    #classify_image(x, model, save=True, name='random_post.pdf')
    #classify_image(x, model, save=True, name='jpmzemke_post.pdf')
    classify_image(x, model)
    plt.show()

    # plot differences of all three channels
    difference = x - img
    plt.matshow(difference[0, :, :, 0])
    plt.title('difference in channel 0')
    plt.matshow(difference[0, :, :, 1])
    plt.title('difference in channel 1')
    plt.matshow(difference[0, :, :, 2])
    plt.title('difference in channel 2')
    plt.show()
