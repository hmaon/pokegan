import numpy as np
import time

#import plaidml.keras
#plaidml.keras.install_backend()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, MaxPool2D
from keras.layers import LeakyReLU, Dropout, ReLU, Concatenate, Add
from keras.layers import BatchNormalization
from keras.layers import Softmax
from keras.layers import CuDNNLSTM, LSTMCell, RNN, GRUCell
from keras.optimizers import Adam, RMSprop, Nadam, Adamax
from keras.callbacks import ModelCheckpoint
from keras import Input, Model

from keras.preprocessing.image import ImageDataGenerator

import keras
import random,math

import PIL
import sys
import math

from scipy import ndimage

EPOCHS = 5000
TRAINATONCE = 4 # pixels to feed in one minibatch

METRICS=[keras.metrics.categorical_accuracy]

start_epoch = 1
load_weights=None

def weights_file(epoch):
    return 'weights-{}.hdf5'.format(epoch)

for i in range(len(sys.argv)):
    if sys.argv[i] == '-epoch':
        start_epoch = int(sys.argv[i+1]) # for convenience when resuming
        load_weights = weights_file(start_epoch)
        start_epoch += 1

###
# D A T A S E T
###

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0,
        height_shift_range=0,
        fill_mode='nearest',
        cval=255,
        horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
        './dir_per_class',
        target_size=(64, 64),
        batch_size=2,
        class_mode='sparse') # batch_size here just means how many images to load at once. for training, a batch is one image  

num_classes = train_generator.num_classes


def gen_RNN():
    #lstm = CuDNNLSTM(6, stateful=True, return_sequences=True)(inp)
    #lstm = LeakyReLU()(lstm)
    #lstm = CuDNNLSTM(3, stateful=True,   return_sequences=True)(lstm)    
    #lstm = Activation('sigmoid')(lstm)
    
    input = Input(batch_shape=(1,1,64*64*3 + num_classes) )
    
    lstm = Concatenate()([Dense(16, activation='softmax', name='zoinks')(input), input])
    cells = []
    for i in range(2):
        cells.append(LSTMCell(256, recurrent_dropout=0.05, implementation=2, recurrent_initializer='glorot_normal'))
    lstm = RNN(cells, stateful=True, name="RNN_yoooo")(lstm)
    lstm = Dense(128, activation='tanh', name='jinkies')(lstm)
    lstm = Dense(3, activation='sigmoid', name='yikes')(lstm)
    lstm = Reshape((1,3,))(lstm)
    model = Model(inputs=input, outputs=lstm)
    
    return model

test_arr = np.concatenate([np.zeros(64*64*3)-1,np.ones(num_classes)*1.0]).reshape((1,1,64*64*3 + num_classes))
    
lstm = gen_RNN()
if load_weights != None:
    lstm.load_weights(load_weights)
lstm.compile(loss='mse', optimizer=Nadam())
lstm.summary()

#predict_lstm = gen_RNN(False)
#predict_lstm.compile(loss='mse', optimizer=Nadam())
#predict_lstm.summary()
#predict_lstm = lstm # ONLINE TRAINING, FUCK IT

#print(lstm.predict( test_arr ))
#print(lstm.predict( test_arr ))
#print(lstm.predict( test_arr ))
#print(lstm.predict( test_arr ))
ptest = lstm.predict( test_arr )
print(ptest.shape)
print(ptest)
print(lstm.predict( test_arr ))
print(lstm.predict( test_arr ))

def predict_image():
    #predict_lstm.predict(np.array((random.uniform(0.,1.), random.uniform(0.,1.), random.uniform(0.,1.))).reshape((1,1,3))) # prime with bullshit
    #predict_lstm.predict(np.array((random.uniform(0.,1.), random.uniform(0.,1.), random.uniform(0.,1.))).reshape((1,1,3))) # prime with bullshit
    pixels = np.zeros(64 * 64 * 3 + num_classes)
    pixels[-1] = 0.95
    for i in range(64*64):
        out = lstm.predict(pixels.reshape((1,1,len(pixels)))).reshape((3,))
        pixels[i:i+3] = out
    img = pixels[:(64*64*3)].reshape((64,64,3))
    img = PIL.Image.fromarray(np.uint8(img*255))
    return img

    
def sample(filenum=0):
    #predict_lstm.set_weights(lstm.get_weights())
    #predict_lstm.compile(optimizer='sgd', loss='mse') # not sure if needed? settings here irrelevant
    #predict_lstm.reset_states()
    lstm.reset_states()
    pad = 3
    swatchdim = 64 # 64x64 is the output of the generator
    swatches = 1 # per side
    dim = (pad+swatchdim) * swatches
    img = PIL.Image.new("RGB", (dim, dim), "white")

    for i in range(swatches * swatches):
        swatch = predict_image()
        sys.stdout.write(str(i) + ' ')
        sys.stdout.flush()
        x = i % swatches
        y = math.floor(i/swatches)
        #print((x,y))
        img.paste(swatch, (x * (pad+swatchdim), y * (pad+swatchdim)))

    img.save('out%d.png' %(filenum,))


#sample(0)
#exit()


###
# T R A I N
###

batches = math.floor(train_generator.n / train_generator.batch_size)

for epoch in range(start_epoch,EPOCHS+start_epoch):
    print("--- epoch %d ---" % (epoch,))
    for batch_num in range(batches):
        lstm.reset_states()

        # get real data
        xs,ys = train_generator.next()
        ys = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) * random.uniform(0.9, 1.0) for cls in ys]) # sparse to one-hot with noising
               
        for x,y in zip(xs,ys):
            print("x: ", x.shape, x[0].shape, x[0].sum(), abs(x[0].sum() - 64*3))
            print("y: ", y.shape, y)

            actual = x
            partial = np.concatenate( [np.zeros(len(actual.reshape((-1,)))) - 1, y] )  # all -1s, then class vector
            losses = []
            for srcY in range(64):
                sys.stdout.write(".")
                sys.stdout.flush()
                for srcX in range(64):
                    _y = np.array((-1., -1., -1.)) if srcY == 63 and srcX == 63 else actual[srcY][srcX]
                    #print("_y: ", _y)
                    
                    # rebuild array with expanded pixels
                    loss = lstm.train_on_batch(partial.reshape((1,1,len(partial))), _y.reshape((1,1,3)))
                    losses.append(loss)
                    a = actual[srcY][srcX]
                    i = (srcY*64 + srcX) * 3
                    partial[i:i+3] = a
                    #print("loss: {}".format(loss))            
            print("\nlosses {}, \nmean {}, median {}".format(losses, np.mean(losses), np.median(losses)))
        
    #train_generator.reset()
    lstm.save(weights_file(epoch))
    sample(epoch)
    #if epoch % 5 == 0:
    #    gen.save('gen-weights-%d.hdf5' % (epoch,))
    
