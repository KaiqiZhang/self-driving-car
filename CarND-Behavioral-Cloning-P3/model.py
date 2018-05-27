import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense, Lambda, Cropping2D
from keras.utils import Sequence
from keras.utils.vis_utils import plot_model

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
STEER_DELTA = 0.2

def get_nvidia_model():
    # model from NVIDIA paper
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(24, kernel_size=(5,5), padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(36, kernel_size=(5,5), padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(48, kernel_size=(5,5), padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu', strides=(1,1)))
    model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu', strides=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.summary()
    return model

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for batch_start in range(0, num_samples, batch_size):
            images = []
            measurements = []
            batch_samples = samples[batch_start:batch_start+batch_size]

            for batch_sample in batch_samples:
                current_path = '../../data/'
                # random camera
                camera = random.choice(['front', 'left', 'right'])
                if camera == 'front':
                    current_path += batch_sample[0].strip()
                    measurement = float(batch_sample[3])
                elif camera == 'left':
                    current_path += batch_sample[1].strip()
                    measurement = float(batch_sample[3]) + STEER_DELTA
                elif camera == 'right':
                    current_path += batch_sample[2].strip()
                    measurement = float(batch_sample[3]) - STEER_DELTA

                image = cv2.imread(current_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # random flip
                flip = random.choice([False, True])
                if flip:
                    images.append(cv2.flip(image_rgb, 1))
                    measurements.append(measurement * -1.0)
                else:
                    images.append(image_rgb)
                    measurements.append(measurement)

            x_train = np.array(images)
            y_train = np.array(measurements)

            yield x_train, y_train

def visualize(lines):
    measurements = []
    for line in lines:
        measurement = float(line[3])
        measurements.append(measurement)

    plt.hist(measurements, bins=80)
    plt.title("Steering angle distribution")
    plt.show()

if __name__ == '__main__':
    # get data
    lines = []
    with open('../../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        lines = [line for line in reader][1:]

    # visualize dataset
    #visualize(lines)

    # split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    # get model
    model = get_nvidia_model()
    #plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(loss='mse', optimizer='adam')

    # train model
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_samples)//BATCH_SIZE,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples)//BATCH_SIZE,
                        epochs=5)

    # save model to disk
    model.save('model.h5')
