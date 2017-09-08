'''
Defines and trains model
'''
import cv2
import csv
import argparse
import numpy as np
import keras.layers

from os import path
from collections import namedtuple

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras import __version__ as keras_version
from pprint import pformat

def get_image(image_path):
    '''
    1. Load image from file path.
    2. Remove top 48 rows (assumption: input image is of size 320x160).
    3. Resize image to 1/8th its current size (to 40x14)
    4. Make sure it is in RGB format
    '''
    image = cv2.imread(image_path)
    assert image is not None
    height, width = image.shape[:2]
    assert height == 160 and width == 320
    # Crop to (112, 320)
    image = image[48:,:]
    # Resize to (14, 40)
    height, width = image.shape[:2]
    assert height == 112 and width == 320
    scale=1/8
    image = cv2.resize(image, (int(width*scale), int(height*scale)), 
                                interpolation = cv2.INTER_AREA)
    assert image is not None
    height, width = image.shape[:2]
    assert height == 14 and width == 40
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image is not None
    return image


def batch_generator(samples, batch_size=32):
    '''
    Given some samples generate some batches. The samples argument is a tuple of
    two parameters. The first parameter is a path to a sample file and the
    second parameter is the desired steering angle at that frame/prosition.
    '''
    num_samples = len(samples)
    
    # Start with a half-sized batch and later double it by adding flipped images
    batch_seed_size = batch_size // 2
    assert batch_seed_size
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_seed_size):
            batch_seed = samples[offset:offset+batch_seed_size]

            images = []
            steering_angles = []
            for sample in batch_seed:
                image = get_image(sample[0])
                images.append(image)
                steering_angles.append(sample[1])
                
                # Add flipped image too
                image = np.fliplr(image)
                images.append(image)
                steering_angles.append(-sample[1])

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)


def get_training_data(data_folders):
    '''
    Collect training data from a list of folders. Each folder should have the 
    same structure and format as the Udacity supplied sample data.
    '''
    samples = []
    for data_folder in data_folders:
        driving_log = path.join(data_folder, "driving_log.csv")
        with open(driving_log) as csvfile:
            reader = csv.reader(csvfile)
            for row in list(reader)[1:]: # Skip the header row
                # Add steering adjust to left and right images
                for img_idx, adj in enumerate([0.0, 0.2, -0.2]):
                    p = row[img_idx].strip()
                    bn = path.basename(p)
                    pd = path.basename(path.dirname(p))
                    img_path = path.join(data_folder, pd, bn)
                    steering_angle = float(row[3])
                    samples.append((img_path, steering_angle + adj))
    
    # Shuffling samples here will make sure data from any given folder does 
    # not have any undue influence because of the order in which it was 
    # included.
    return shuffle(samples)
            
            
def train(model, training_data, epochs=10, batch_size=32):
    '''
    Train the given model with the given training data.
    '''    
    training_samples, validation_samples = train_test_split(training_data, test_size=0.2)
    
    training_batches = batch_generator(training_samples, batch_size=batch_size)
    validation_batches = batch_generator(validation_samples, batch_size=batch_size)
    
    steps_per_epoch = len(training_samples)/batch_size
    validation_steps = len(validation_samples)/batch_size
    
    model.fit_generator(training_batches, 
                        steps_per_epoch=steps_per_epoch, 
                        epochs=epochs, 
                        validation_data=validation_batches,
                        validation_steps=validation_steps)
    

def get_model():
    '''
    Get the Keras model for this project.
    '''
    input_image = Input(shape=(14,40,3))
    
    # Normalize the input image to facilitate model fitting.
    normalized = Lambda(lambda x: (x / 255.0) - 0.5)(input_image)
    
    # Feed the normalized image to a Conv2D layer for feature extraction.
    conv = Conv2D(filters=32, kernel_size=5, activation='relu')(normalized)
    
    # Feed the convolved image to another Conv2D layer for higher level 
    # feature extraction.
    conv = Conv2D(filters=32, kernel_size=5, activation='relu')(conv)
    
    # Flatten the convolved output for use in the fully connected layers.
    flattened_conv = Flatten()(conv)
    
    # Fully connected layers with dropout
    fc = Dense(120, activation='elu')(flattened_conv)
    fc = Dense(80, activation='elu')(fc)
    fc = Dense(20, activation='elu')(fc)
    output_steering_angle = Dense(1)(fc)
    
    model = Model(
                inputs=[input_image], 
                outputs=[output_steering_angle]
            )
    # Using the 'mean squared error' loss function with the Adam optimizer.
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def format_dict(dict):
    return ', '.join(["{}={}".format(k,v) for k,v in dict.items()])


def get_args():
    parser = argparse.ArgumentParser(description='Train and save the keras model for the behavioral cloning project.')
    parser.add_argument('-f', '--data-folder', nargs='*')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=5)
    args = parser.parse_args()
    assert args.data_folder
    return args
    
    
def main(args):
    print("Args:", format_dict(vars(args)))
    print("Keras version:", keras_version)
    model = get_model()
    training_data = get_training_data(args.data_folder)
    train(model, training_data, args.epochs, args.batch_size)
    model.save('model.h5')


if __name__ == '__main__':
    main(get_args())
