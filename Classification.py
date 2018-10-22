import os
import cv2
import json
import shutil
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()


def mkdir():
    if not os.path.exists('./output'):
        os.makedirs('./output')
    if not os.path.exists('./output/red'):
        os.makedirs('./output/red')
    if not os.path.exists('./output/green'):
        os.makedirs('./output/green')
    if not os.path.exists('./output/blue'):
        os.makedirs('./output/blue')
    if not os.path.exists('./output/other'):
        os.makedirs('./output/other')


def preProcessing(name):
    img = cv2.imread(name) / 255  # Normalized image the required entry interval is [0,1]
    w, h, c = img.shape
    size = w * h
    img = img[..., ::-1]  # This one is funny BGR->RGB
    img = cv2.resize(img, (10, 10))  # Resize to accelerate speed
    img = img.reshape(-1, 3)  # Shape 2D image into 1D vector
    return img, size


def compute(img):
    colors = tf.constant(img, dtype=tf.float32)
    model = tf.keras.models.model_from_json(json.load(open("model.json"))["model"], custom_objects={})
    model.load_weights("model_weights.h5")
    predictions = model.predict(colors, batch_size=32, verbose=0)
    # Output is one-hot vector for 9 class:["red","green","blue","orange","yellow","pink", "purple","brown","grey"]
    predictions = tf.one_hot(np.argmax(predictions, 1), 9)
    # Sum along the column, each entry indicates no of pixels
    res = tf.reduce_sum(predictions, reduction_indices=0).numpy()
    # Threshold is 0.5 (accuracy is 96%) change threshold may cause accuracy decrease
    if res[0] / (sum(res[:-1]) + 1) > 0.5:
        return "red"
    elif res[1] / (sum(res[:-1]) + 1) > 0.5:
        return "green"
    elif res[2] / (sum(res[:-1]) + 1) > 0.5:
        return "blue"
    else:
        return "other"


def copy(name, full_name, size, label):
    if label == 'red':
        shutil.copy(full_name, "./output/red")
        os.rename('./output/red/' + name, './output/red/' + str(size) + '.png')
    if label == 'green':
        shutil.copy(full_name, "./output/green")
        os.rename('./output/green/' + name, './output/green/' + str(size) + '.png')
    if label == 'blue':
        shutil.copy(full_name, "./output/blue")
        os.rename('./output/blue/' + name, './output/blue/' + str(size) + '.png')
    if label == 'other':
        shutil.copy(full_name, "./output/other")
        os.rename('./output/other/' + name, './output/other/' + str(size) + '.png')


def sort():
    red = "./output/red/"
    path = os.listdir(red)
    nums = [int(x[:-4]) for x in path]
    nums.sort()
    i = 1
    for num in nums:
        os.rename('./output/red/' + str(num) + '.png', './output/red/' + str(i) + '_' + str(num) + '.png')
        i += 1

    green = "./output/green/"
    path = os.listdir(green)
    nums = [int(x[:-4]) for x in path]
    nums.sort()
    i = 1
    for num in nums:
        os.rename('./output/green/' + str(num) + '.png', './output/green/' + str(i) + '_' + str(num) + '.png')
        i += 1

    blue = "./output/blue/"
    path = os.listdir(blue)
    nums = [int(x[:-4]) for x in path]
    nums.sort()
    i = 1
    for num in nums:
        os.rename('./output/blue/' + str(num) + '.png', './output/blue/' + str(i) + '_' + str(num) + '.png')
        i += 1

    other = "./output/other/"
    path = os.listdir(other)
    nums = [int(x[:-4]) for x in path]
    nums.sort()
    i = 1
    for num in nums:
        os.rename('./output/other/' + str(num) + '.png', './output/other/' + str(i) + '_' + str(num) + '.png')
        i += 1


def main():
    files = os.listdir('./input/')
    for name in files:
        mkdir()
        full_name = './input/' + name
        img, size = preProcessing(full_name)
        label = compute(img)
        copy(name, full_name, size, label)
    sort()


main()
