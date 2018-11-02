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


def gen_RNN(training):
    #lstm = CuDNNLSTM(6, stateful=True, return_sequences=True)(inp)
    #lstm = LeakyReLU()(lstm)
    #lstm = CuDNNLSTM(3, stateful=True,   return_sequences=True)(lstm)    
    #lstm = Activation('sigmoid')(lstm)
    

    input = Input(batch_shape=( (TRAINATONCE if training else 1) , 1, 3 + num_classes) )
    
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

test_arr = np.concatenate([np.array( (0.,0.,1.) ),np.ones(num_classes)*1.0]).reshape(1,1,3 + num_classes)
    
lstm = gen_RNN(True)
if load_weights != None:
    lstm.load_weights(load_weights)
lstm.compile(loss='mse', optimizer=Nadam())
lstm.summary()

predict_lstm = gen_RNN(False)
predict_lstm.compile(loss='mse', optimizer=Nadam())
predict_lstm.summary()
#predict_lstm = lstm # ONLINE TRAINING, FUCK IT

#print(lstm.predict( test_arr ))
#print(lstm.predict( test_arr ))
#print(lstm.predict( test_arr ))
#print(lstm.predict( test_arr ))
ptest = predict_lstm.predict( test_arr )
print(ptest.shape)
print(ptest)
print(predict_lstm.predict( test_arr ))
print(predict_lstm.predict( test_arr ))

def predict_image():
    #predict_lstm.predict(np.array((random.uniform(0.,1.), random.uniform(0.,1.), random.uniform(0.,1.))).reshape((1,1,3))) # prime with bullshit
    #predict_lstm.predict(np.array((random.uniform(0.,1.), random.uniform(0.,1.), random.uniform(0.,1.))).reshape((1,1,3))) # prime with bullshit
    c = np.zeros(num_classes)
    c[1] =.95
    predict_lstm.predict(np.concatenate([np.array((random.uniform(0.,1.), random.uniform(0.,1.), random.uniform(0.,1.))),c]).reshape((1,1,3+num_classes))) # prime with bullshit    
    pixels = [np.array((-1., -1., -1.))] # our special start pixel ???
    #pixels = [np.array((1.,1.,1.))]
    for i in range(64*64):
        pixels.append(predict_lstm.predict(np.concatenate([pixels[-1], c]).reshape((1,1,3+num_classes)))[0][0] )
    img = np.array(pixels[-(64*64):]).reshape((64,64,3))
    img = PIL.Image.fromarray(np.uint8(img*255))
    return img

    
def sample(filenum=0):
    predict_lstm.set_weights(lstm.get_weights())
    #predict_lstm.compile(optimizer='sgd', loss='mse') # not sure if needed? settings here irrelevant
    predict_lstm.reset_states()
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
            print(x.shape, x[0].shape, x[0].sum(), abs(x[0].sum() - 64*3))
            while abs(x[0].sum()  - 64*3) < 0.01 or abs(x[0].sum()) < 0.01:
                print("parp")
                x = np.delete(x, 0, axis=0)

            while abs(x[-1].sum()  - 64*3) < 0.01 or abs(x[-1].sum()) < 0.01:
                print("bluh")
                x = np.delete(x, -1, axis=0)
                
            print(x.shape)
            x[0][0][0] = x[0][0][1] = x[0][0][2] = -1 # introduce special start pixel ?
            mbatches = x.reshape((-1,TRAINATONCE,1,3))
            print("next image, class={}, reshaped={}".format(y, mbatches.shape))
            losses = []
            for i in range(len(mbatches)):
                _x = mbatches[i]
                #print(_x)
                _y = np.roll(_x, -1)
                _y[-1] = mbatches[(i+1) % len(mbatches)][0]
                #_y = mbatches[(i+1) % len(mbatches)]
                if abs(_x.sum() - _y.sum()) < 0.01:
                    continue
                #print(_x, _y)
                
                # rebuild array with expanded pixels
                new = []
                for __x in _x:
                    #print(_x, __x)
                    timestep = []
                    new.append(timestep)
                    timestep.append(np.concatenate([__x[0], y]))
                new = np.array(new)
                loss = lstm.train_on_batch(new.reshape((TRAINATONCE,1,3+num_classes)), _y.reshape((TRAINATONCE,1,3)))
                losses.append(loss)
                #print("loss: {}".format(loss))            
            print("losses {}, mean {}, median {}".format(losses, np.mean(losses), np.median(losses)))
        
    #train_generator.reset()
    lstm.save(weights_file(epoch))
    sample(epoch)
    #if epoch % 5 == 0:
    #    gen.save('gen-weights-%d.hdf5' % (epoch,))
    
