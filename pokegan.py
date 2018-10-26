import numpy as np
import time

import plaidml.keras
plaidml.keras.install_backend()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D
from keras.layers import LeakyReLU, Dropout, ReLU
from keras.layers import BatchNormalization
from keras.layers import Softmax
from keras.optimizers import Adam, RMSprop, Nadam

from keras.preprocessing.image import ImageDataGenerator

import keras

import matplotlib.pyplot as plt

EPOCHS = 25

METRICS=[keras.metrics.categorical_accuracy]

###
# D A T A S E T
###

train_datagen = ImageDataGenerator(
        rescale=1./255,
        #width_shift=10,
        #height_shift=10,
        #fill_mode='constant',
        #cval=255,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        './dir_per_class',
        target_size=(96, 96),
        batch_size=64,
        class_mode='binary')

num_classes = train_generator.num_classes + 1 # pokemon + fake

###
# D I S C R I M I N A T O R
####


def d_block(model, separables = 128, pointwise = 64, stride=2):
    model.add(SeparableConv2D(separables, 3, strides=stride, padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2D(pointwise,1,kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

def Discriminator():
    discrim = Sequential()
    discrim.add(Conv2D(64, 3, padding='same', input_shape=(96,96,3), kernel_initializer='glorot_normal'))
    discrim.add(BatchNormalization())
    discrim.add(LeakyReLU())

    d_block(discrim, 128, 64)

    d_block(discrim, 256, 128)

    d_block(discrim, 512, 256, 2)
    d_block(discrim, 512, 256, 2)
    #d_block(discrim, 512, 256, 2)
    d_block(discrim, 512, 256, 2)
    d_block(discrim, 512, 512, 2)

    discrim.add(Flatten())

    discrim.add(Dense(num_classes*2))
    discrim.add(BatchNormalization())
    discrim.add(LeakyReLU())

    discrim.add(Dense(num_classes*2))
    discrim.add(BatchNormalization())    
    discrim.add(LeakyReLU())

    discrim.add(Dense(num_classes, activation='tanh'))
    discrim.add(Softmax())

    return discrim

discrim = Discriminator()

discrim.summary()

discrim.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=METRICS)

###
# G E N E R A T O R
###

def Generator():
    gen = keras.Sequential()
    gen.add(Dense(12*12*64, input_shape=(100+653,)))
    gen.add(Reshape(target_shape=(12,12,64)))
    gen.add(ReLU())
    for i in range(4):
        gen.add(Conv2DTranspose(32, 3, padding='same', strides=2))
        gen.add(BatchNormalization())
        gen.add(ReLU())
    gen.add(Conv2D(3, 1))
    gen.summary()
    return gen

gen = Generator()
gen.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=METRICS)

###
# adversarial model
###
for layer in discrim.layers:
    layer.trainable = false

adver = Sequential()
adver.add(gen)
adver.add(discrim)
adver.compile(loss='categorical_crossentropy', metrics=METRICS)


###
# T R A I N
###

for epoch in range(1,EPOCHS+1):
    print("--- epoch %d ---" % (epoch,))
    while train_generator.total_batches_seen * train_generator.batch_size < train_generator.n * epoch:
        x,y = train_generator.next()
        d_loss = discrim.train_on_batch(x, keras.utils.to_categorical(y, num_classes=num_classes))
        print("d_loss %f, %s %f " % (d_loss[0], discrim.metrics_names[1], d_loss[1]))
    #train_generator.reset()
