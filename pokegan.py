import numpy as np
import time

#import plaidml.keras
#plaidml.keras.install_backend()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, MaxPool2D
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

from scipy import ndimage

EPOCHS = 5000

METRICS=[keras.metrics.categorical_accuracy]

###
# D A T A S E T
###

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=5,
        height_shift_range=5,
        fill_mode='constant',
        cval=255,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        './dir_per_class',
        target_size=(96, 96),
        batch_size=16,
        class_mode='binary')

num_classes = train_generator.num_classes + 1 # pokemon + fake

###
# D I S C R I M I N A T O R
####


def d_block(model, separables = 128, depth_multiplier=1, stride=1, maxpool=True):
    model.add(SeparableConv2D(separables, 3, strides=stride,
                              depth_multiplier=depth_multiplier,
                              padding='same',
                              depthwise_initializer='glorot_normal',
                              pointwise_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if maxpool:
        model.add(MaxPool2D(padding='same'))
    #model.add(Conv2D(pointwise,1,kernel_initializer='glorot_normal'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU())

def Discriminator():
    discrim = Sequential()
    discrim.add(Conv2D(64, 3, padding='same', input_shape=(96,96,3), kernel_initializer='glorot_normal'))
    discrim.add(BatchNormalization())
    discrim.add(LeakyReLU())

    d_block(discrim, 128, 1)

    d_block(discrim, 256, 1)

    d_block(discrim, 512, 1)
    d_block(discrim, 512, 1)
    #d_block(discrim, 512, 256, 2)
    d_block(discrim, 512, maxpool=False)
    d_block(discrim, 512, maxpool=False)

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

discrim.compile(optimizer=Adam(lr = 0.002), loss='categorical_crossentropy', metrics=METRICS)

###
# G E N E R A T O R
###

def g_block(model, depth=32, stride=2, size=3):
    model.add(Conv2DTranspose(32, size, padding='same', strides=stride))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    

def Generator():
    gen = keras.Sequential()
    gen.add(Dense(6*6*256*3, input_shape=(100+num_classes,)))
    gen.add(Reshape(target_shape=(6,6,256*3)))
    gen.add(BatchNormalization())
    g_block(gen, 256*3)
    g_block(gen, 128*3)
    g_block(gen, 64*3)
    g_block(gen, 32*3)
    g_block(gen, depth=32*3, size=5, stride=1)
    g_block(gen, depth=32*3, stride=1)
    g_block(gen, depth=32*3, stride=1)
    gen.add(Conv2DTranspose(3, 1, activation='sigmoid'))
    gen.summary()
    return gen

gen = Generator()
gen.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=METRICS)

# _class is one-hot category array
# randomized if None
def gen_input(_class=None):
    noise = np.random.uniform(-2.0, 2.0, 100).clip(-1.0,1.0) # partially saturated noise to encourage features to be fully on or off?
    if type(_class) == type(None):
        _class = keras.utils.to_categorical(random.randint(0, num_classes-2), num_classes=num_classes) * random.uniform(0.9,1.0)
    return np.concatenate((_class, noise))

def gen_input_rand():
    return gen_input()

# optionally receives one-hot class array from training loop
def gen_input_batch(classes=None):
    if type(classes) != type(None):
        return np.array([gen_input(cls) for cls in classes])
    print("!!! Generating random batch in gen_input_batch()!")
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
        

#sample(666)
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
        # get real data
        x,y = train_generator.next()
        y = real_y = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) * random.uniform(0.9, 1.0) for cls in y]) # sparse to one-hot with noising
        d_loss = discrim.train_on_batch(x, y)
        print("REAL: d_loss %f, %s %f " % (d_loss[0], discrim.metrics_names[1], d_loss[1]))
        
        # get fake data
        x_gen_input = gen_input_batch(real_y) # real classes with appended random noise inputs
        x = gen.predict(x_gen_input)
        y = np.array([keras.utils.to_categorical(num_classes-1, num_classes = num_classes) for dummy in x])
        d_loss = discrim.train_on_batch(x, y)         
        print("FAKE: d_loss %f, %s %f " % (d_loss[0], discrim.metrics_names[1], d_loss[1]))
        
        #x = gen_input_batch() 
        #y = np.array([inp[:num_classes] for inp in x])
        a_loss = adver.train_on_batch(x_gen_input, real_y)
        print("ADVR: a_loss %f, %s %f " % (a_loss[0], adver.metrics_names[1], a_loss[1]))
        x_gen_input = gen_input_batch(real_y) # same classes, different noise
        a_loss = adver.train_on_batch(x_gen_input, real_y)
        print("ADVR: a_loss %f, %s %f @ %d/%d\n" % (a_loss[0], adver.metrics_names[1], a_loss[1], batch_num, batches))
        
    #train_generator.reset()
    sample(epoch)
    if epoch % 5 == 0:
        gen.save('gen-weights-%d.hdf5' % (epoch,))
        discrim.save('discrim-weights-%d.hdf5' % (epoch,))
    
