import numpy as np
import time

#import plaidml.keras
#plaidml.keras.install_backend()

import tensorflow as tf
#tf.enable_eager_execution()



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, MaxPool2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, ReLU, PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Softmax, Input, Concatenate, Add
from tensorflow.train import AdamOptimizer, RMSPropOptimizer, ProximalAdagradOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras

import random,math
import sys,os
from timeit import default_timer as timer

import PIL

from scipy import ndimage

from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
set_session(tf.Session(config=config))

ctx = None
    
if 'mxnet' == K.backend():
    import mxnet
    ctx = ["gpu(0)"]
    
    

#keras.backend.set_floatx('float16')
    
EPOCHS = 5000

METRICS=[keras.metrics.categorical_accuracy]

INIT='glorot_normal'

GEN_WEIGHTS="gen-weights-{}.hdf5"
DISCRIM_WEIGHTS="discrim-weights-{}.hdf5"


load_disc=None
load_gen=None
start_epoch=1

tags = ["A", "B", "C", "D"]

for i in range(len(sys.argv)):
    if sys.argv[i] == '-epoch':
        start_epoch = int(sys.argv[i+1]) # for convenience when resuming
        tag = tags[ start_epoch % len(tags) ]
        load_disc = DISCRIM_WEIGHTS.format(tag)
        load_gen = GEN_WEIGHTS.format(tag)
        start_epoch += 1
    if sys.argv[i] == '-d':
        load_disc = sys.argv[i+1]
    if sys.argv[i] == '-g':
        load_gen = sys.argv[i+1]


###
### D A T A S E T
###

def rangeshift(inp):
    inp = (inp * -1) + 127
    #print(inp)
    return inp

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=3,
        height_shift_range=3,
        fill_mode='nearest',
        cval=255,
        horizontal_flip=False,
        data_format='channels_first')

train_generator = train_datagen.flow_from_directory(
        './dir_per_class',
        target_size=(64, 64),
        batch_size=60,
        class_mode='sparse',
        interpolation='lanczos')

num_classes = train_generator.num_classes + 1 # pokemon + fake

###
### D I S C R I M I N A T O R
###

def d_block(dtensor, depth = 128, stride=1, maxpool=False):

    # feature detection
    dtensor = SeparableConv2D(depth, 3, strides=1,\
                              padding='same',\
                              kernel_initializer=INIT)(dtensor)
    dtensor = PReLU()(dtensor)
    dtensor = BatchNormalization()(dtensor)
    
    # strided higher level feature detection
    dtensor = SeparableConv2D(depth, 3, strides=stride,\
                              padding='same',\
                              kernel_initializer=INIT)(dtensor)
    dtensor = PReLU()(dtensor)
    dtensor = BatchNormalization()(dtensor)
    if maxpool:
        dtensor = MaxPool2D(padding='same')(dtensor)
        
    # nonsense?
    dtensor = Conv2D(depth, 1, kernel_initializer=INIT)(dtensor)
    dtensor = PReLU()(dtensor)
    dtensor = BatchNormalization()(dtensor)    
    return dtensor

# unused
def res_d_block(dtensor, depth, stride = 1):
    short = dtensor
    
    conv = d_block(dtensor, depth, stride)
    conv = d_block(conv, depth, 1)

    # diagram at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 ... seems weird but who am I to argue
    if stride == 2:
        short = AveragePooling2D()(short)
    short = PReLU()(short)  
    short = BatchNormalization()(short)    
    short = SeparableConv2D(depth, 1)(short) 

    short = PReLU()(short)
    short = BatchNormalization()(short)    
    short = SeparableConv2D(depth, 1)(short)
    short = PReLU()(short)
    short = BatchNormalization()(short)    
    
        
    return Add()([conv, short])
    
def Discriminator():
    dense_dropout = 0.05
    
    inp = Input((3,64,64))
    
    
    # fine feature discrimation in two full conv layers?
    d = SeparableConv2D(32, 5, padding='same', input_shape=(3,64,64), kernel_initializer=INIT)(inp)
    BatchNormalization()(d)
    PReLU()(d)
    
    d = SeparableConv2D(64, 3, padding='same', kernel_initializer=INIT, strides=2)(d)
    BatchNormalization()(d)
    PReLU()(d)
    
    # 32x32 here
    d = d_block(d, 256, 2) # 16x16
    d = d_block(d, 512, 2) # 8x8
    d = d_block(d, 512, 2) # 4x4
    d = d_block(d, 512, 2) # 2x2
    
    e = MaxPooling2D()(d)
    e = Flatten()(e)
    
    d = Conv2D(512, 1, kernel_initializer=INIT)(d) 
    d = PReLU()(d)
    d = BatchNormalization()(d)
    
    d = Flatten()(d)
    d = Concatenate()([d,e])

    # classify ??
    d = Dense(512, kernel_initializer=INIT)(d)
    d = Dropout(dense_dropout)(d)
    d = PReLU()(d)
    d = BatchNormalization()(d)
    
    d = Dense(512, kernel_initializer=INIT)(d)
    d = Dropout(dense_dropout)(d)
    d = PReLU()(d)
    d = BatchNormalization()(d)    
    
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

discrim.compile(optimizer=RMSPropOptimizer(learning_rate=0.001), loss='kullback_leibler_divergence', metrics=METRICS)

###
### G E N E R A T O R
###

def g_block(gtensor, depth=32, stride=1, size=3, upsample=True):
    conv = gtensor
    if upsample: 
        conv = UpSampling2D()(conv)

    #conv = SeparableConv2D(depth, 3, depth_multiplier=2, padding='same', kernel_initializer=INIT)(conv)
    conv = Conv2D(depth, 3, padding='same', kernel_initializer=INIT)(conv)
    conv = PReLU()(conv)  
    conv = BatchNormalization()(conv)
        
    #conv = Conv2DTranspose(depth, size, padding='same', strides=stride, kernel_initializer=INIT)(conv)
    #conv = PReLU()(conv)     
    #conv = BatchNormalization()(conv)    

    #conv = Conv2D(depth, 1, padding='same', kernel_initializer=INIT)(conv)
    #conv = PReLU()(conv)  
    #conv = BatchNormalization()(conv)    
    
    return conv
    
NOISE = 50
    
def Generator():
    input = Input((NOISE+num_classes,))
    #g = Dense(512, kernel_initializer=INIT)(input)    
    #g = PReLU()(g)
    #g = BatchNormalization()(g)

    g = Dense(4*4*512, kernel_initializer=INIT)(input)
    g = PReLU()(g)
    g = BatchNormalization()(g)
    
    g = Reshape(target_shape=(512,4,4))(g)

    g = g_block(g, 512, 2, upsample=True) # 8x8
    
    g = g_block(g, 256, 2, upsample=True) # 16x16
    
    g = g_block(g, 128, 2, size=5, upsample=True)  # 32x32
    
    g = g_block(g, 64, 2, size=5, upsample=True) # 64x64

    # I don't know what these are supposed to do but whatever:
    
    g = g_block(g, 64, 1, size=3, upsample=False) # 64x64
    
    #g = SeparableConv2D(256, 3, depth_multiplier=2, padding='same', kernel_initializer=INIT)(g)
    #g = PReLU()(g)
    #g = BatchNormalization()(g)

    #g = Conv2DTranspose(256, 3, padding='same', kernel_initializer=INIT)(g)
    #g = PReLU()(g)
    #g = BatchNormalization()(g)
    
    #g = Conv2D(1024, 1, padding='same', kernel_initializer=INIT)(g)
    #g = PReLU()(g)
    #g = BatchNormalization()(g)
    
    g = Conv2D(3, 1, activation='sigmoid')(g)
    g = Reshape((3, 64, 64))(g) # not sure if needed but we're doing channels_first; it helps as a sanity check when coding, at least!
    
    gen = Model(inputs=input, outputs=g)
    gen.summary()
    return gen

gen = Generator()

if load_gen and os.path.isfile(load_gen):
    gen.load_weights(load_gen)
else:
    print("not loading weights for generator")


# _class is one-hot category array
# randomized if None
def gen_input(_class=None):
    noise = np.random.uniform(0.0, 1.0, NOISE)
    if type(_class) == type(None):
        _class = keras.utils.to_categorical(random.randint(0, num_classes-2), num_classes=num_classes) * 0.95
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
    pad = 3
    swatchdim = 64 # 64x64 is the output of the generator
    swatches = 5 # per side
    dim = (pad+swatchdim) * swatches
    img = PIL.Image.new("RGB", (dim, dim), "white")

    for i in range(min(swatches * swatches, len(all_out))):
        out = all_out[i]
        out = out.reshape(3, 64, 64)
        out = np.uint8(out * 255)
        out = np.moveaxis(out, 0, -1) # switch from channels_first to channels_last
        #print("check this: ")
        #print(out.shape)
        swatch = PIL.Image.fromarray(out)
        x = i % swatches
        y = math.floor(i/swatches)
        #print((x,y))
        img.paste(swatch, (x * (pad+swatchdim), y * (pad+swatchdim)))

    img.save('out%d.png' %(filenum,))
    
def sample(filenum=0):
    all_out = []
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 128, 129, 130, 131, 132, 133, 134, 135, 136, 119, 101, 93, 142, num_classes-1, 143]
    print(len(classes))
    _classidx=0
    for i in range(5):
        inp = gen_input_batch([keras.utils.to_categorical(classes[x] % num_classes, num_classes=num_classes) for x in range(_classidx,_classidx+5)])
        _classidx+=5
        print(inp.shape)
        batch_out = gen.predict(inp)
        print(batch_out.shape)
        for out in batch_out:
            all_out.append(out)
    render(all_out, filenum)
            
        

###
### adversarial model
###
#for layer in discrim.layers:
#    layer.trainable = False

discrim.trainable = False

adver = Sequential()
adver.add(gen)
adver.add(discrim)
adver.summary()

adver.compile(optimizer=RMSPropOptimizer(learning_rate=0.001), loss='kullback_leibler_divergence', metrics=METRICS) 
gen.compile(optimizer=RMSPropOptimizer(learning_rate=0.001), loss='kullback_leibler_divergence', metrics=METRICS)

### some instrumentation
#render(train_generator.next()[0], -99)
sample(0)
train_generator.reset()
#exit()


###
### ugh
###

def get_lr(model):
    return 0
    return float(keras.backend.eval(model.optimizer.lr))

def set_lr(model, newlr):
    pass
    #model.optimizer.lr = keras.backend.variable(newlr)
    
###
### T R A I N
###

batches = math.floor(train_generator.n / train_generator.batch_size)

disc_start_lr = get_lr(discrim)
disc_real_lr = get_lr(discrim)
disc_fake_lr = get_lr(discrim)

adver_start_lr = get_lr(adver)
adver_lr = get_lr(adver)

batches_timed = 0
total_time = 0

for epoch in range(start_epoch,EPOCHS+1):
    print("--- epoch %d ---" % (epoch,))
    for batch_num in range(batches):
        start = timer()
        # get real data
        x,y = train_generator.next()
        if len(y) != train_generator.batch_size:
            continue # avoid re-analysis of ops due to changing batch sizes
        y = real_y = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) * 0.95 for cls in y])
        set_lr(discrim, disc_real_lr)
        d_loss = discrim.train_on_batch(x, y)
        print("REAL: d_loss %f, %s %f " % (d_loss[0], discrim.metrics_names[1], d_loss[1]))
        disc_real_lr = abs(min(1.0, d_loss[0])) * disc_start_lr
        
        # get fake data
        #half_y = real_y[math.floor(len(real_y)/2):]
        x_gen_input = gen_input_batch(real_y) # real classes with appended random noise inputs
        x = gen.predict(x_gen_input)
        y = np.array([keras.utils.to_categorical(num_classes-1, num_classes = num_classes) * 0.95 for dummy in x])
        set_lr(discrim, disc_fake_lr)
        d_loss = discrim.train_on_batch(x, y)         
        print("FAKE: d_loss %f, %s %f " % (d_loss[0], discrim.metrics_names[1], d_loss[1]))
        disc_fake_lr = abs(min(1.0, d_loss[0])) * disc_start_lr
        
        #x = gen_input_batch() 
        #y = np.array([inp[:num_classes] for inp in x])
        set_lr(adver, adver_lr)
        a_loss = adver.train_on_batch(x_gen_input, real_y)
        print("ADVR: a_loss %f, %s %f " % (a_loss[0], adver.metrics_names[1], a_loss[1]))
        x_gen_input = gen_input_batch(real_y) # same classes, different noise
        a_loss = adver.train_on_batch(x_gen_input, real_y)
        print("ADVR: a_loss %f, %s %f @ %d/%d\n" % (a_loss[0], adver.metrics_names[1], a_loss[1], batch_num, batches))
        adver_lr = abs(min(2.0, a_loss[0])) * adver_start_lr
        end = timer()
        batches_timed += 1
        total_time += (end - start)
        print("batch time: {}, total_time: {}, mean time: {}\n".format(end-start, total_time, total_time / batches_timed))
        
    #train_generator.reset()
    sample(epoch)
    if epoch % 5 == 0:
        tag = tags[epoch % len(tags)]
        gen.save(GEN_WEIGHTS.format(tag))
        discrim.save(DISCRIM_WEIGHTS.format(tag))
    
