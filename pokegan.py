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
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

import keras
import random,math

import matplotlib.pyplot as plot
import PIL

EPOCHS = 5000

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
        batch_size=36,
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

def g_block(model, stride=2):
    model.add(Conv2DTranspose(32, 3, padding='same', strides=stride))
    model.add(BatchNormalization())
    model.add(ReLU())
    

def Generator():
    gen = keras.Sequential()
    gen.add(Dense(12*12*64, input_shape=(100+653,)))
    gen.add(Reshape(target_shape=(12,12,64)))
    gen.add(ReLU())
    for i in range(3):
        g_block(gen)
    g_block(gen,1)
    gen.add(Conv2D(3, 1, activation='sigmoid'))
    gen.summary()
    return gen

gen = Generator()
gen.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=METRICS)

def gen_input(_class=0):
    noise = np.random.uniform(-1.0, 1.0, 100)
    return np.concatenate((keras.utils.to_categorical(_class, num_classes=num_classes), noise))

def gen_input_rand():
    return gen_input(random.randint(0, num_classes-1))

def gen_input_batch():
    return np.array([gen_input_rand() for x in range(train_generator.batch_size)])
        
def sample(filenum=0):
    inp = gen_input_batch()
    print(inp.shape)
    batch_out = gen.predict(inp)
    print(batch_out.shape)
    dim = math.ceil(math.sqrt(train_generator.batch_size))
    fig, plots = plot.subplots(dim, dim)
    plot.tight_layout()
    plots = plots.reshape(len(plots) * len(plots[0]))
    for i in range(len(batch_out)):
        out = batch_out[i]
        out = out.reshape(96, 96, 3)
        out = np.uint8(out*255)
        img = PIL.Image.fromarray(out)
        plots[i].imshow(img)
        plots[i].axis('off')

    plot.savefig('out%d.png' %(filenum,))
        

#sample()
#exit()

###
# adversarial model
###
for layer in discrim.layers:
    layer.trainable = False

adver = Sequential()
adver.add(gen)
adver.add(discrim)
adver.summary()
adver.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=METRICS)


###
# T R A I N
###

batches = math.floor(train_generator.n / train_generator.batch_size)

for epoch in range(1,EPOCHS+1):
    print("--- epoch %d ---" % (epoch,))
    for batch_num in range(batches):
        # get fake data
        x = gen.predict(gen_input_batch())
        y = np.array([keras.utils.to_categorical(num_classes-1, num_classes = num_classes) for dummy in x])
        # get real data
        real_x,real_y = train_generator.next()
        real_y = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) for cls in real_y])
        x = np.concatenate((x, real_x))
        y = np.concatenate((y, real_y))
        d_loss = discrim.train_on_batch(x, y)        
        print("d_loss %f, %s %f " % (d_loss[0], discrim.metrics_names[1], d_loss[1]))
        x = gen_input_batch()
        y = np.array([inp[:num_classes] for inp in x])
        a_loss = adver.train_on_batch(x, y)
        print("a_loss %f, %s %f @ %d/%d" % (a_loss[0], adver.metrics_names[1], a_loss[1], batch_num, batches))
    #train_generator.reset()
    sample(epoch)
    gen.save('gen-weights-%d.hdf5' % (epoch,))
    discrim.save('discrim-weights-%d.hdf5' % (epoch,))
    
