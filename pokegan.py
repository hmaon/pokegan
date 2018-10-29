import numpy as np
import time

#import plaidml.keras
#plaidml.keras.install_backend()

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, MaxPool2D, AveragePooling2D
from keras.layers import LeakyReLU, Dropout, ReLU
from keras.layers import BatchNormalization
from keras.layers import Softmax, Input, Concatenate, Add
from keras.optimizers import Adam, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

import keras
import random,math
import sys,os

import matplotlib.pyplot as plot
import PIL

from scipy import ndimage

EPOCHS = 5000

METRICS=[keras.metrics.categorical_accuracy]

INIT='glorot_normal'

GEN_WEIGHTS="gen-weights-%d.hdf5"
DISCRIM_WEIGHTS="discrim-weights-%d.hdf5"


load_disc=None
load_gen=None
start_epoch=1

for i in range(len(sys.argv)):
    if sys.argv[i] == '-epoch':
        start_epoch = int(sys.argv[i+1]) # for convenience when resuming
        load_disc = DISCRIM_WEIGHTS % (start_epoch,)
        load_gen = GEN_WEIGHTS % (start_epoch,)
        start_epoch += 1
    if sys.argv[i] == '-d':
        load_disc = sys.argv[i+1]
    if sys.argv[i] == '-g':
        load_gen = sys.argv[i+1]


###
### D A T A S E T
###

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=3,
        height_shift_range=3,
        fill_mode='constant',
        cval=255,
        horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
        './dir_per_class',
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse',
        interpolation='lanczos')

num_classes = train_generator.num_classes + 1 # pokemon + fake

###
### D I S C R I M I N A T O R
###

def d_block(dtensor, separables = 128, depth_multiplier=1, stride=1, maxpool=False):
    dtensor = SeparableConv2D(separables, 3, strides=stride,\
                              depth_multiplier=depth_multiplier,\
                              padding='same',\
                              depthwise_initializer=INIT,\
                              pointwise_initializer=INIT)(dtensor)
    dtensor = BatchNormalization()(dtensor)
    dtensor = LeakyReLU()(dtensor)
    if maxpool:
        dtensor = MaxPool2D(padding='same')(dtensor)
    dtensor = Conv2D(separables * depth_multiplier, 1, kernel_initializer=INIT)(dtensor)
    dtensor = BatchNormalization()(dtensor)
    dtensor = LeakyReLU()(dtensor)
    return dtensor

def res_d_block(dtensor, depth, depth_multiplier, stride = 1):
    short = dtensor
    
    conv = d_block(dtensor, depth, depth_multiplier, stride)
    conv = d_block(conv, depth, depth_multiplier, 1)

    # diagram at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 ... seems weird but who am I to argue
    if stride == 2:
        short = AveragePooling2D()(short)
    short = BatchNormalization()(short)
    short = LeakyReLU()(short)  
    short = Conv2D(depth * depth_multiplier, 1)(short) # kernel size of stride is probably wrong

    short = BatchNormalization()(short)
    short = LeakyReLU()(short)    
    short = Conv2D(depth * depth_multiplier, 1)(short)
        
    return Add()([conv, short])
    
def Discriminator():
    dense_dropout = 0.1
    
    inp = Input((64,64,3))
    
    
    # fine feature discrimation in two full conv layers?
    d = Conv2D(64, 7, padding='same', input_shape=(64,64,3), kernel_initializer=INIT, strides=2)(inp)
    BatchNormalization()(d)
    LeakyReLU()(d)
    
    # 32x32 here
    
    # strided depthwise separable convolutions to classify high level features
    d = res_d_block(d, 64, 3, stride=2) # 16x16
    #d = res_d_block(d, 64, 3)
    d = res_d_block(d, 64, 3)

    d = res_d_block(d, 128, 3, stride=2) # 8x8
    #d = res_d_block(d, 128, 3)
    d = res_d_block(d, 128, 3)

    d = res_d_block(d, 256, 3, stride=2) # 4x4
    #d = res_d_block(d, 256, 3)
    d = res_d_block(d, 256, 3)

    d = res_d_block(d, 512, 3, stride=2) # 2x2
    #d = res_d_block(d, 512, 3)
    d = res_d_block(d, 512, 3)    

    # smush feature maps?
    d = Conv2D(1024, 1, kernel_initializer=INIT)(d) 
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    
    d = Flatten()(d)

    # classify ??
    d = Dense(num_classes*3, kernel_initializer=INIT)(d)
    d = BatchNormalization()(d)
    d = Dropout(dense_dropout)(d)
    d = LeakyReLU()(d)
    
    d = Dense(num_classes, kernel_initializer=INIT)(d)
    d = Softmax()(d)
        
    discrim = Model(inputs=inp, outputs=d)    

    return discrim

discrim = Discriminator()

discrim.summary()

if load_disc and os.path.isfile(load_disc):
    discrim.load_weights(load_disc)
else:
    print("not loading weights for discriminator")

discrim.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=METRICS)

###
### G E N E R A T O R
###

def g_block(gtensor, depth=32, stride=1, size=3, upsample=True):
    conv = gtensor
    if upsample: 
        conv = UpSampling2D()(conv)
    
    conv = Conv2DTranspose(depth, size, padding='same', strides=stride, kernel_initializer=INIT)(conv)
    conv = BatchNormalization()(conv)    
    conv = LeakyReLU()(conv)     
    
    conv = Conv2D(depth, 1, padding='same', kernel_initializer=INIT)(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)  
    
    return conv
    
NOISE = num_classes*2
    
def Generator():
    input = Input((NOISE+num_classes,))
    g = Dense(4*4*NOISE, kernel_initializer=INIT)(input)
    g = BatchNormalization()(g)
    g = LeakyReLU()(g)

    g = Dense(4*4*NOISE, kernel_initializer=INIT)(input)
    g = BatchNormalization()(g)
    g = LeakyReLU()(g)
    
    g = Reshape(target_shape=(4,4,NOISE))(g)
    
    g = g_block(g, 512, 2, upsample=False) # 8x8  
    
    h = g_block(g, 256, 2, upsample=False) # 16x16
    g = g_block(g, 256) # 16x16
    
    h = g_block(h, 128, 2, upsample=False)  # 32x32
    g = g_block(g, 128)  # 32x32
    
    h = g_block(h, 64, 2, size=5, upsample=False) # 64x64
    g = g_block(g, 64, size=5)  # 64x64
    g = Concatenate()([g, h])

    # I don't know what these are supposed to do but whatever:
    g = Conv2D(256, 1, padding='same', kernel_initializer=INIT)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU()(g)

    g = Conv2D(256, 1, padding='same', kernel_initializer=INIT)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU()(g)
    
    g = Conv2D(3, 1, activation='sigmoid')(g)
    
    gen = Model(inputs=input, outputs=g)
    gen.summary()
    return gen

gen = Generator()

if load_gen and os.path.isfile(load_gen):
    gen.load_weights(load_gen)
else:
    print("not loading weights for generator")

gen.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=METRICS)

# _class is one-hot category array
# randomized if None
def gen_input(_class=None):
    noise = np.random.uniform(-2.0, 2.0, NOISE).clip(-1.0,1.0) # partially saturated noise to encourage features to be fully on or off?
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

def render(all_out, filenum=0):            
    dim = math.ceil(math.sqrt(len(all_out)))
    fig, plots = plot.subplots(dim, dim)
    plot.tight_layout(h_pad=3)
    plots = plots.reshape(len(plots) * len(plots[0]))
    for i in range(len(all_out)):
        out = all_out[i]
        out = out.reshape(64, 64, 3)
        out = np.uint8(out*255)
        img = PIL.Image.fromarray(out)
        img.resize((128,128))
        plots[i].imshow(img)
        plots[i].axis('off')
        #plots[i].get_yaxis().set_visible(False)
        #plots[i].set_xlabel(str(i+1))

    plot.savefig('out%d.png' %(filenum,), dpi=400)
    
def sample(filenum=0):
    all_out = []
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 128, 129, 130, 131, 132, 133, 134, 135, 136, 119, 101, 93, 142, num_classes-1, 143]
    print(len(classes))
    _classidx=1
    for i in range(16):
        inp = gen_input_batch([keras.utils.to_categorical(classes[x], num_classes=num_classes) for x in range(_classidx,_classidx+4)])
        _classidx+=4
        print(inp.shape)
        batch_out = gen.predict(inp)
        print(batch_out.shape)
        for out in batch_out:
            all_out.append(out)
    render(all_out, filenum)
            
        
render(train_generator.next()[0], -99)
sample(0)
train_generator.reset()
#exit()

###
### adversarial model
###
for layer in discrim.layers:
    layer.trainable = False

adver = Sequential()
adver.add(gen)
adver.add(discrim)
adver.summary()
adver.compile(optimizer=RMSprop(lr = 0.005), loss='categorical_crossentropy', metrics=METRICS)


###
### T R A I N
###

batches = math.floor(train_generator.n / train_generator.batch_size)

for epoch in range(start_epoch,EPOCHS+1):
    print("--- epoch %d ---" % (epoch,))
    for batch_num in range(batches):
        # get real data
        x,y = train_generator.next()
        if len(y) != train_generator.batch_size:
            continue # avoid re-analysis of ops due to changing batch sizes
        y = real_y = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) * random.uniform(0.9, 1.0) for cls in y])
        d_loss = discrim.train_on_batch(x, y)
        print("REAL: d_loss %f, %s %f " % (d_loss[0], discrim.metrics_names[1], d_loss[1]))
        
        # get fake data
        #half_y = real_y[math.floor(len(real_y)/2):]
        x_gen_input = gen_input_batch(real_y) # real classes with appended random noise inputs
        x = gen.predict(x_gen_input)
        y = np.array([keras.utils.to_categorical(num_classes-1, num_classes = num_classes) * random.uniform(0.9,1.0) for dummy in x])
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
        gen.save(GEN_WEIGHTS % (epoch,))
        discrim.save(DISCRIM_WEIGHTS % (epoch,))
    
