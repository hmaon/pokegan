import numpy as np
import queue
import threading
import time
import math


# TODO:
# * implement tf.train.Checkpoint
# * float16 ops XXX not functioning
# XXX implement stabilization from Roth et al. (2017) https://arxiv.org/abs/1705.09367 XXX
# * SELU!

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, GaussianNoise
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, MaxPool2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, ReLU, PReLU, ELU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Softmax, Input, Concatenate, Add
from tensorflow.train import AdamOptimizer, RMSPropOptimizer, ProximalAdagradOptimizer, GradientDescentOptimizer
from tensorflow.contrib.opt import NadamOptimizer, AdamWOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.constraints import MinMaxNorm

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras

import tensorflow.contrib.eager as tfe

import random,math
import sys,os
from timeit import default_timer as timer

import PIL

from scipy import ndimage

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 12,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'dir_per_class',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16. [XXX NOT WORKING]""")
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.85,
                            """Fraction of GPU memory to reserve; 1.0 may crash the OS window manager.""")
tf.app.flags.DEFINE_integer('epoch', 1,
                            """Epoch number of last checkpoint for continuation! (e.g., number of last screenshot)""")
tf.app.flags.DEFINE_string('gen', None,
                           """Generator weights to load.""")
tf.app.flags.DEFINE_string('disc', None,
                           """Discriminator weights to load.""")
                           
batch_size = FLAGS.batch_size
                           
dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
npdtype = np.float16 if FLAGS.use_fp16 else np.float32

data_format=tf.keras.backend.image_data_format()
channels_shape = (3,64,64) if data_format == 'channels_first' else (64,64,3)
                            
from tensorflow.keras.backend import set_session
config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
session = tf.Session(config=config)
set_session(session)

ctx = None

save_path = 'summary/'

global_step = tf.train.create_global_step()
writer = tf.contrib.summary.create_file_writer(save_path)
        
#keras.backend.set_floatx('float16')
    
EPOCHS = 5000
d_learning_base_rate = 0.001
g_learning_base_rate = 0.001
weight_decay = 1e-5

#METRICS=[keras.metrics.categorical_accuracy]

#INIT=tf.initializers.orthogonal(dtype = dtype)
INIT=tf.initializers.glorot_normal(dtype = dtype)
#INIT=tf.initializers.lecun_normal()
PRELUINIT=keras.initializers.Constant(value=0.3)
#g_batchnorm_constraint = MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=[0])
g_batchnorm_constraint = None

#K_REG=tf.keras.regularizers.l2(weight_decay)
K_REG=None

GEN_WEIGHTS="gen-weights-{}.hdf5"
DISCRIM_WEIGHTS="discrim-weights-{}.hdf5"

#VIRTUAL_BATCH_SIZE=batch_size
RENORM=False
BATCH2D_AXIS=1

load_disc=None
load_gen=None
start_epoch=FLAGS.epoch

tags = ["A", "B", "C", "D"]

if start_epoch > 1:
    tag = tags[ math.floor(start_epoch/4) % len(tags) ]
    load_disc = DISCRIM_WEIGHTS.format(tag)
    load_gen = GEN_WEIGHTS.format(tag)
    start_epoch += 1

if FLAGS.disc != None:
    load_disc = FLAGS.disc
elif FLAGS.disc == 'None':
    load_disc = None
    
if FLAGS.gen != None:
    load_gen = FLAGS.gen
elif FLAGS.disc == 'None':
    load_disc = None
    
#for i in range(len(sys.argv)):
#    if sys.argv[i] == '-d':
#        load_disc = sys.argv[i+1]
#    if sys.argv[i] == '-g':
#        load_gen = sys.argv[i+1]


###
### D A T A S E T
###

input_noise_scale = 0.2 * (0.995 ** start_epoch)

def prepro(inp):
    inp += np.random.normal(scale = input_noise_scale * 127., size=inp.shape)
    inp = inp - 127.
    inp = np.clip(inp, -127., 127.)
    return inp

with tf.device('/cpu:0'): 
    ImageDatagen = ImageDataGenerator(
            rescale=1./127,
            width_shift_range=8,
            height_shift_range=8,
            zoom_range=.02,
            fill_mode='wrap',
            cval=255,
            horizontal_flip=False,
            data_format='channels_first',
            preprocessing_function=prepro,
            dtype=dtype)

    train_gen = ImageDatagen.flow_from_directory(
            FLAGS.data_dir,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='sparse',
            interpolation='lanczos')

    num_classes = train_gen.num_classes

    flowQueue = queue.Queue(10)

    class LoadThread(threading.Thread):
        def __init__(self, q, datagen):
            threading.Thread.__init__(self)
            self.q = q
            self.datagen = datagen

        def run(self):
            print ("Starting image data loader thread")
            while True:
                while self.q.full():
                    time.sleep(.05)
                self.q.put(self.datagen.next())

    def next_batch():
        while flowQueue.empty():
            time.sleep(.05)
        ret = flowQueue.get()
        return ret

    loadThread = LoadThread(flowQueue, train_gen)
    loadThread.daemon = True
    loadThread.start()

    def img_float_to_uint8(img, do_reshape=True):
        out = img
        if do_reshape: out = out.reshape(channels_shape)
        out = np.uint8(out * 127)
        return out
        
    def img_uint8_to_float(img):
        img = dtype(img)
        img *= 1./512
        return img

    # save a grid of images to a file
    def render(all_out, filenum=0):            
        pad = 3
        swatchdim = 64 # 64x64 is the output of the generator
        swatches = 6 # per side
        dim = (pad+swatchdim) * swatches
        img = PIL.Image.new("RGB", (dim, dim), "white")

        for i in range(min(swatches * swatches, len(all_out))):
            out = all_out[i]
            out = out + 1.0
            #out = tf.clip_by_value(out, 0., 2.).numpy()
            out = np.clip(out, 0., 2.)
            out = img_float_to_uint8(out)
            #print("pre-move shape ", out.shape)
            out = np.moveaxis(out, 0, -1) # switch from channels_first to channels_last
            #print("check this: ", out.shape)
            swatch = PIL.Image.fromarray(out)
            x = i % swatches
            y = math.floor(i/swatches)
            #print((x,y))
            img.paste(swatch, (x * (pad+swatchdim), y * (pad+swatchdim)))

        img.save('out%d.jpeg' %(filenum,))
        #ima = np.array(img).flatten()
        #ima = ima[:batch_size*swatchdim*swatchdim*3]
        #print("ima.shape: {}".format(ima.shape))
        #with writer.as_default():
        #    with tf.contrib.summary.always_record_summaries():
        #        tf.contrib.summary.image("sample", ima.reshape((batch_size,swatchdim,swatchdim,3)))

#test rendering and loading
render(next_batch()[0], -99999)
print("sample render done")
    
###
### D I S C R I M I N A T O R
###

def d_block(dtensor, depth = 128, stride=1, maxpool=False, stridedPadding='same', size=3):

    # feature detection
    dtensor = Conv2D(depth, size, strides=1,\
                              padding='same', use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    dtensor = LeakyReLU(0.2)(dtensor)
    dtensor = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
    
    # strided higher level feature detection
    dtensor = Conv2D(depth, size, strides=stride,\
                              padding=stridedPadding, use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    dtensor = LeakyReLU(0.2)(dtensor)
    dtensor = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
    if maxpool:
        dtensor = MaxPool2D(padding='same')(dtensor)
        
    # nonsense?
    #dtensor = Conv2D(depth, 1, kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    #dtensor = LeakyReLU(0.2)(dtensor)
    #dtensor = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)    
    return dtensor

# ResNet-like implementation
def res_d_block(dtensor, depth, stride = 1, stridedPadding='same', extra=3):

    if stride > 1:
        dtensor = Conv2D(depth, 3, strides=stride,\
                                  padding=stridedPadding,\
                                  kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(dtensor)
        dtensor = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
        dtensor = LeakyReLU(0.2)(dtensor)

    short = dtensor

    for i in range(extra):
        dtensor = Conv2D(depth, 3, strides=1,\
                                  padding='same',\
                                  kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(dtensor)
        dtensor = LeakyReLU(0.2)(dtensor)
        dtensor = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
    
    # diagram at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 ... re: full pre-activation, sort of
    for i in range(2):
        short = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(short)    
        short = LeakyReLU(0.2)(short)
            
        short = Conv2D(depth, 1, strides=1,\
                      padding='same',\
                      kernel_initializer='orthogonal', kernel_regularizer=K_REG, use_bias=False)(short)
            
    dtensor = Add()([dtensor, short])
    return dtensor

# semi-DenseNet-like block    
def dense_d_block(dtensor, depth = 128, stride=1, stridedPadding='same', size=3, extra=4):

    maps = []

    dt2 = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)            
    dt2 = LeakyReLU(0.2)(dt2)

    # strided higher level feature detection
    dtensor = Conv2D(depth, size, strides=stride,\
                              padding=stridedPadding, use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    
    for i in range(extra):
        dt2 = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)            
        dt2 = LeakyReLU(0.2)(dt2)
        
        dt2 = Conv2D(depth, size, strides=1,\
                                  padding=stridedPadding, use_bias=False,\
                                  kernel_initializer=INIT, kernel_regularizer=K_REG)(dt2)

        dtensor = Concatenate(axis=1)([dtensor, dt2])            
        
    dtensor = Concatenate(axis=1)(maps)        
    return dtensor

    

#gaussian_rate = tfe.Variable(d_learning_base_rate)    # can't include in Discriminator() because .save() crashes on it :( :( :( wtf :(

def Discriminator():
    dense_dropout = 0.25
    
    inp = Input(channels_shape)    
    
    d = inp
        
    d = Conv2D(64, 7, padding='same', kernel_initializer=INIT, kernel_regularizer=K_REG, strides=1, use_bias=False)(d) # 64x64
    d = LeakyReLU(0.2)(d)          
    d = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, )(d)
    
    d = res_d_block(d, 64, 2) # 32x32
    d = res_d_block(d, 64, 1) # 32x32
    d = res_d_block(d, 64, 1) # 32x32

    d = res_d_block(d, 128, 2) # 16x16
    d = res_d_block(d, 128, 1) # 16x16
    d = res_d_block(d, 128, 1) # 16x16
    
    d = res_d_block(d, 256, 2) # 8x8
    d = res_d_block(d, 256, 1) # 8x8
    d = res_d_block(d, 256, 1) # 8x8

    d = res_d_block(d, 256, 2, extra=3) # 4x4        
    d = res_d_block(d, 256, 1, extra=3) # 4x4        
    d = res_d_block(d, 256, 1, extra=3) # 4x4        
    
    d = res_d_block(d, 256, 2) # 2x2
    d = res_d_block(d, 256, 1) # 2x2
    d = res_d_block(d, 256, 1) # 2x2

    d = Flatten()(d)
    
    #d = Concatenate()([d, a])
    
    d = Dense(1024, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(d)
    d = LeakyReLU(0.2)(d)
    d = Dropout(dense_dropout)(d)
    d = BatchNormalization(renorm=RENORM, )(d)

    e = d
    
    # classify ??    
    d = Dense(num_classes, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(d) # classifier
    #d = Softmax()(d) # calculated in softmax_cross_entropy() but not in hinge_loss()?
    

    e = Dense(128, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(e)
    e = LeakyReLU(0.2)(e)
    e = BatchNormalization(renorm=RENORM, )(e)
    e = Dense(1, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(e) # realness
    
    discrim = Model(inputs=inp, outputs=[d,e]) # class, realness    

    return discrim

discrim = Discriminator()

discrim.summary()

if load_disc and os.path.isfile(load_disc):
    discrim.load_weights(load_disc)
else:
    print("not loading weights for discriminator")


discrim.call = tf.contrib.eager.defun(discrim.call)



# from https://github.com/rothk/Stabilizing_GANs
# D1 = disc_real, D2 = disc_fake
# can't get this to work but only tried before correctly implementing ACGAN (if I ever did)
def Discriminator_Regularizer(D1_logits, D1_arg, D2_logits, D2_arg, tape=None):
    #D1 = tf.nn.sigmoid(D1_logits)
    #D2 = tf.nn.sigmoid(D2_logits)
    D1 = tf.nn.softmax(D1_logits)
    D2 = tf.nn.softmax(D2_logits)

    # convnert real real to max probability of any real sample
    D1 = D1[:,:-1]
    D1 = tf.convert_to_tensor([(max(x),) for x in D1], dtype=dtype)
    
    # convert fake prediction 1 - probability of fakeness
    D2 = 1 - D2[:,-1:]
    #D2 = D2[:,-1:]
    # or not? it doesn't work and I don't get why yet
    
    #print("D1 = {}, D2 = {}".format(D1, D2))
    
    if tape:
        # eager version
        with tape.stop_recording():
            grad_D1_logits = tape.gradient(D1_logits, D1_arg)#[0]
            #print(grad_D1_logits.shape)
            #grad_D1_logits = grad_D1_logits[0]
            #print(grad_D1_logits.shape)
            grad_D2_logits = tape.gradient(D2_logits, D2_arg)#[0]
    else:
        grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
        grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [batch_size,-1]), axis=BATCH2D_AXIS, keep_dims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [batch_size,-1]), axis=BATCH2D_AXIS, keep_dims=True)

    #grad_D1_logits_norm = tf.tile(grad_D1_logits_norm, (1, num_classes))
    #grad_D2_logits_norm = tf.tile(grad_D2_logits_norm, (1, num_classes))
    #print("{} vs {}".format(grad_D1_logits_norm.shape, D1.shape))
    #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    assert grad_D1_logits_norm.shape == D1.shape
    assert grad_D2_logits_norm.shape == D2.shape

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer

#Discriminator_Regularizer = tf.contrib.eager.defun(Discriminator_Regularizer) # fails!
    
###
### G E N E R A T O R
###

# semi-dense generator block...
def g_block(gtensor, depth=32, stride=1, size=3, upsample=False, deconvolve=True, extra=3):
    conv = gtensor
    if upsample: 
        conv = UpSampling2D(dtype=dtype)(conv)
    
    if deconvolve:
        conv = Conv2DTranspose(depth, size, padding='same', strides=stride, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(conv)
            
    if extra > 0:
        # maps = [conv] # closer to true DenseNet architecture
        maps = [] # lighter-weight version forcing all deconv output through conv2ds
                      
        for i in range(extra):
            if len(maps) > 1:         
                conv = Concatenate(axis=1)(maps)

            conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=[2,3])(conv)    
            conv = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)        
                
            conv = Conv2D(depth, size, padding='same', kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(conv)
            maps.append(conv)

        conv = Concatenate(axis=1)(maps)
        conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=[2,3])(conv)    
        conv = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)        
            
    return conv
    
NOISE = 100
    
def Generator():
    xz = Input((num_classes+NOISE,), dtype=dtype)    
    
    g = xz
    
    #zc = Reshape(target_shape=(-1, 1, 1))(xz)
    #zc = UpSampling2D()(zc) # 2x2
    #zc = UpSampling2D()(zc) # 4x4
    #zc = UpSampling2D()(zc) # 8x8

    #g = Dense(512, kernel_initializer=INIT, kernel_regularizer=K_REG)(g)
    #g = PReLU(alpha_initializer=PRELUINIT)(g)
    #g = BatchNormalization(renorm=RENORM, )(g)

    g = Dense(2*2*1024, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(g)
    g = PReLU(alpha_initializer=PRELUINIT)(g)
    g = BatchNormalization(renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(g)
    
    #q = Dense(64*64*3, kernel_initializer=INIT, kernel_regularizer=K_REG, dtype=dtype, activation='tanh')(g)
    #q = Reshape(channels_shape)(q)
    
    #g = Reshape(target_shape=(2,2,1024))(g)
    g = Reshape(target_shape=(-1,2,2))(g)

    #g = g_block(g, 512, 2)
    #print("2x2 shape? ", g.shape)
    g = g_block(g, 512, 2) # 4x4
    
    g = g_block(g, 256, 2) # 8x8

    #zc = UpSampling2D()(zc) # 16x16
    
    g = g_block(g, 128, 2) # 16x16

    #g = Concatenate(axis=1)([g,zc])

    #zc = UpSampling2D()(zc) # 32x32
    
    #q = Dense(16*16*4, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(xz)
    #q = PReLU(alpha_initializer=PRELUINIT)(q)
    #q = BatchNormalization(renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(q)    
    #q = Reshape((-1, 16, 16))(q)
    
    #g = Concatenate(axis=1)([g, q])
    
    g = g_block(g, 64, 2)  # 32x32
    
    
    #g = Concatenate()([g,zc])
    
    
    g = g_block(g, 32, 2) # 64x64

    #g = Conv2D(64, 3, padding='same', kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(g)
    #g = PReLU(alpha_initializer=PRELUINIT, shared_axes=[2,3])(g)    
    #g = BatchNormalization(axis=BATCH2D_AXIS, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(g)        
    
    #print(g.shape)
    g = Conv2D(3, 1, activation='tanh', padding='same', use_bias=False)(g)    
    g = Reshape(channels_shape)(g) 
    
    gen = Model(inputs=xz, outputs=g)
    gen.summary()
    return gen

gen = Generator()

if load_gen and os.path.isfile(load_gen):
    gen.load_weights(load_gen)
    print(gen.layers[1].weights)
else:
    print("not loading weights for generator")

gen.call = tf.contrib.eager.defun(gen.call)



# _class is one-hot category array
# randomized if None
def gen_input(_class=None, clipping=0.95):
    #noise = np.random.uniform(-clipping, clipping, NOISE)
    #noise = np.random.permutation(NOISE) * (1.0 / NOISE) % clipping
    noise = np.random.normal(loc = 0, scale=1, size=NOISE)
    noise = (noise % clipping) * (np.abs(noise) / noise)
    if type(_class) == type(None):
        _class = keras.utils.to_categorical(random.randint(0, num_classes-2), num_classes=num_classes)
    return np.concatenate((_class, noise))

def gen_input_rand(clipping=1.0):
    return gen_input(clipping)

# optionally receives one-hot class array from training loop
def gen_input_batch(classes=None, clipping=1):
    if type(classes) != type(None):
        return np.array([gen_input(cls, clipping) for cls in classes], dtype=npdtype)
    print("!!! Generating random batch in gen_input_batch()!")
    return tf.convert_to_tensor(np.array([gen_input_rand(clipping) for x in range(batch_size)], dtype=npdtype))

        
def sample(filenum=0):
    all_out = []
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 128, 129, 130, 131, 132, 133, 134, 135, 136, 119, 101, 93, 142, num_classes-1, 143]
    #print("[sample()] num classes: ", len(classes))
    _classidx=0
    for i in range(4):
        inp = gen_input_batch([keras.utils.to_categorical(classes[x] % num_classes, num_classes=num_classes) for x in range(_classidx,_classidx+10)], clipping=0.25*(i+1))
        _classidx += 10
        print(inp.shape)
        batch_out = gen.predict(inp)
        print(batch_out.shape)
        for out in batch_out:
            all_out.append(out)
    render(all_out, filenum)
        

###
### optimizer config
###

g_learning_rate = tfe.Variable(g_learning_base_rate)
#gen_optimizer = RMSPropOptimizer(learning_rate=g_learning_rate)
gen_optimizer = AdamOptimizer(learning_rate=g_learning_rate, beta1=0.5)
#gen_optimizer = AdamWOptimizer(learning_rate=g_learning_rate, beta1=0.5, weight_decay = 0.000001)
    
d_learning_rate = tfe.Variable(d_learning_base_rate)
#discrim_optimizer = RMSPropOptimizer(learning_rate=d_learning_rate)    
discrim_optimizer = AdamOptimizer(learning_rate=d_learning_rate, beta1=0.5)
#discrim_optimizer = AdamWOptimizer(learning_rate=d_learning_rate, beta1=0.5, weight_decay = 0.000001)
#discrim_optimizer = GradientDescentOptimizer(d_learning_rate)

#for layer in discrim.layers:
#    layer.trainable = False

#discrim.trainable = False

#adver = Sequential()
#adver.add(gen)
#adver.add(discrim)
#adver.summary()

#adver.compile(optimizer=RMSPropOptimizer(learning_rate=0.002), loss='kullback_leibler_divergence', metrics=METRICS) 

### some instrumentation
sample(0)
#train_generator.reset()
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
###
### T R A I N
###
### T R A I N
###
### T R A I N
###
### C H O O - C H O O
###
###

replayQueue = []

batches = math.floor(train_gen.n / train_gen.batch_size)
if start_epoch > 0:
    global_step.assign(batches * start_epoch)

batches_timed = 0
total_time = 0

# for https://github.com/rothk/Stabilizing_GANs regularizer; 0.1 is a value they used without annealing in one of their examples 
gamma = 0.1

def apply_gen_gradients(gradients_of_generator, gen):
    gen_optimizer.apply_gradients(zip(gradients_of_generator, gen.variables))

apply_gen_gradients = tf.contrib.eager.defun(apply_gen_gradients)

def apply_discrim_gradients(gradients_of_discriminator, discrim):
    discrim_optimizer.apply_gradients(zip(gradients_of_discriminator, discrim.variables))

apply_discrim_gradients = tf.contrib.eager.defun(apply_discrim_gradients)
  
for epoch in range(start_epoch,EPOCHS+1):
    print("--- epoch %d ---" % (epoch,))
    input_noise_scale = 0.2 * (0.995 ** (epoch + flowQueue.maxsize)) # effect of this is delayed due to queue but oh well
    for batch_num in range(batches):
        start = timer()
        
        # get real data
        x,y = next_batch()
        if len(y) != train_gen.batch_size:
            continue # avoid various problems from changes in batch size
        x = tf.convert_to_tensor(x) 
        
        y = real_y = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) for cls in y], dtype=npdtype)
        yG  = np.array([keras.utils.to_categorical(random.randint(0, num_classes-1), num_classes=num_classes) for x in range(batch_size)], dtype=npdtype) # random samples for fake output
        yG2 = np.array([keras.utils.to_categorical(random.randint(0, num_classes-1), num_classes=num_classes) for x in range(batch_size)], dtype=npdtype) # random samples for fake output
        
        yG=tf.convert_to_tensor(yG)
        yG2=tf.convert_to_tensor(yG2)
        
        ### non-class version:
        yG = tf.zeros_like(yG)
        yG2 = tf.zeros_like(yG2)
        
        #weights = tf.ones_like(y) * 0.75 + real_y * (.25/.9) + all_fake_y * (.25/.9)
        #weights = real_y / .9 # applied only to generator; concentrate only on maximizing similarity to real output, not on beating discriminator at fake vs. real game
        
        with tf.GradientTape() as gtape:
            with tf.GradientTape() as dtape:
                dtape.watch(x)
                gtape.watch(x)
                #tape.watch(y)
                #dtape.watch(yG)
                gtape.watch(yG2)
                
                #real_out = discrim(x, training=True)
                #d_loss = tf.losses.softmax_cross_entropy(real_y, real_out)
                #d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_out, labels=y)
                #d_loss = tf.losses.hinge_loss(labels=real_y, logits=real_out)
                #d_loss_real *= d_loss_real # square hinge
                #d_acc = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(real_y, real_out))                
                
                x_gen_input = gen_input_batch(yG) 
                gen_x = gen(x_gen_input, training=True)
                
                if batch_num % 3 == 0:
                    replayQueue.insert(0, gen_x.numpy())
                
                gen_x += np.random.normal(scale=input_noise_scale, size=gen_x.shape)

                d_class,dx_realness = discrim(x, training=True)
                _,df_realness = discrim(gen_x, training=True)               
                
                #dx_realness = tf.reshape(dx_realness, (batch_size,))
                #df_realness = tf.reshape(df_realness, (batch_size,))
                                        
                #d_loss = tf.losses.softmax_cross_entropy(y, d_class)

                #d_loss = tf.losses.hinge_loss(labels = tf.zeros_like(df_realness), logits = df_realness) + tf.losses.hinge_loss(labels=tf.ones_like(dx_realness), logits=dx_realness) + tf.losses.mean_squared_error(tf.zeros_like(df_realness), tf.sigmoid(df_realness)) + tf.losses.mean_squared_error(tf.ones_like(dx_realness), tf.sigmoid(dx_realness))
                
                #d_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(dx_realness), logits = dx_realness - df_realness) + tf.losses.mean_squared_error(tf.zeros_like(df_realness), tf.sigmoid(df_realness)) + tf.losses.mean_squared_error(tf.ones_like(dx_realness), tf.sigmoid(dx_realness))
                #d_loss = (tf.losses.sigmoid_cross_entropy(dx_realness - tf.math.reduce_mean(df_realness), tf.ones_like(dx_realness)) + tf.losses.sigmoid_cross_entropy(df_realness - tf.math.reduce_mean(dx_realness), tf.zeros_like(df_realness)))
                
                d_loss_real = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(dx_realness), logits = dx_realness)
                d_loss_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(df_realness), logits = df_realness)
                
                #d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness), logits = dx_realness - tf.math.reduce_mean(df_realness))
                #d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(df_realness), logits = df_realness - tf.math.reduce_mean(dx_realness))
                                
                #d_loss = d_loss_real + d_loss_fake
                #d_loss /= 2

                d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness) * 1.0, dx_realness - df_realness)
                
                if len(replayQueue) > 30 and batch_num % 1 == 0:
                    random.shuffle(replayQueue)
                    gen_r = replayQueue.pop()
                    _,dreplay_realness = discrim(gen_r, training=True)                           
                    d_loss += tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness) * 1.0, dx_realness - dreplay_realness)
                    d_loss *= .5
                    #d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness) * .99, dx_realness - dreplay_realness)
                
                #d_l2 = tf.add_n(discrim.losses)
                d_l2 = -1
                #d_loss += d_l2 * .01

            
            gradients_of_discriminator = dtape.gradient(d_loss, discrim.variables)
            
            # call regularizer from https://github.com/rothk/Stabilizing_GANs
            #disc_reg = Discriminator_Regularizer(real_out, x, gen_out, gen_x, tape)
            #disc_reg = (gamma/2.0)*disc_reg
            #d_loss += disc_reg

            #df_realness = tf.concat([df_realness[8:], df_realness[:8]], axis=0)
            
            #gen_out_a = out[len(real_y):]
        
            #x_gen_input = gen_input_batch(yG2)  
            #gen_x = gen(x_gen_input, training=True)
            #g_class,g_realness = discrim(gen_x, training=True)

            #g_loss = tf.losses.softmax_cross_entropy(yG2, logits = g_class)
            
            #g_loss = tf.losses.hinge_loss(labels=tf.ones_like(g_realness), logits=g_realness) 
            #g_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(g_realness), logits = (g_realness - dx_realness)) + tf.losses.mean_squared_error(tf.ones_like(df_realness), tf.sigmoid(df_realness))
            #g_loss =  tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(df_realness) * .9, logits = df_realness)
            #g_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dx_realness), logits = dx_realness - tf.math.reduce_mean(df_realness)) + tf.losses.sigmoid_cross_entropy(tf.ones_like(df_realness), logits = df_realness - tf.math.reduce_mean(dx_realness))
            g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness) * 1.0, logits = df_realness - dx_realness)

            #g_loss += tf.losses.mean_squared_error(tf.ones_like(g_realness), tf.sigmoid(g_realness))
            #g_loss += tf.losses.sigmoid_cross_entropy(multi_class_labels = all_real_b, logits = df_realness)
            #g_loss += tf.losses.sigmoid_cross_entropy(multi_class_labels = all_real, logits = g_realness)
            
            g_loss_nonrel = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(df_realness), logits=df_realness)
            
            #gloss = 0
            
            #g_loss /= 2
            
            #g_l2 = tf.add_n(gen.losses)
            g_l2 = -1
            #g_loss += g_l2 * .01
            
        #g_acc = tf.keras.metrics.mean_squared_error(all_real, tf.nn.sigmoid(tf.reshape(g_realness, (-1,))))

        #d_acc_class = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(real_y, tf.nn.softmax(d_class))) / batch_size
        #d_acc_realness = (tf.keras.metrics.mean_squared_error(all_fake, tf.nn.sigmoid(tf.reshape(df_realness, (-1,)))) + (tf.keras.metrics.mean_squared_error(all_real, tf.nn.sigmoid(tf.reshape(dx_realness, (-1,))))) ) * .5            

        apply_discrim_gradients(gradients_of_discriminator, discrim)

        gradients_of_generator = gtape.gradient(g_loss, gen.variables)
        apply_gen_gradients(gradients_of_generator, gen)
                
        end = timer()
        batches_timed += 1
        total_time += (end - start)
        
        #d_loss_real = tf.reduce_mean(d_loss_real)
        #d_loss_fake = tf.reduce_mean(d_loss_fake)
        with writer.as_default():
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("d_loss_real", d_loss)
                tf.contrib.summary.scalar("d_loss_fake", d_loss)
                tf.contrib.summary.scalar("g_loss", g_loss)                
                tf.contrib.summary.scalar("g_loss_nonrel", g_loss_nonrel)                
                #tf.contrib.summary.scalar("d_acc_realness", d_acc_realness)
                #tf.contrib.summary.scalar("g_accuracy", g_acc)
                #tf.contrib.summary.histogram("d_grads?", gradients_of_discriminator[0])
                #tf.contrib.summary.histogram("g_grads?", gradients_of_generator[0]) 
        
        writer.flush()
        
        print("epoch {}; batch {} / {}".format(epoch, batch_num, batches))
        print("dx_realness... {}".format(dx_realness))
        #print("d_loss: {:.7f}, d_l2: {:.7f}".format(d_loss, d_l2)) 
        print("d_loss_real: {:.7f}, d_loss_fake: {:.7f}, combined: {:.7}, d_l2: {:.7f}".format(d_loss_real, d_loss_fake, d_loss, d_l2))
        #print("d_realness MSE: {:.3f}, d_class MSE: {:.3f}".format(d_acc_realness, d_acc_class))
        #print("g_loss: {:.7f}, acc: {:.3f}".format((g_loss), g_acc/batch_size))
        print("g_loss: {:.7f}, g_loss_nonrel {:.7f}, g_l2: {:.7f}".format(g_loss, g_loss_nonrel, g_l2))
        #print("mean d grads: {:.8f}, mean g grads: {:.8f}".format(tf.reduce_mean(gradients_of_generator), tf.reduce_mean(gradients_of_discriminator)))
        print("batch time: {:.3f}, total_time: {:.3f}, mean time: {:.3f}\n".format(end-start, total_time, total_time / batches_timed))
        
        global_step.assign_add(1)

    #with writer.as_default():
    #    with tf.contrib.summary.always_record_summaries():
    #        for w in gen.weights: tf.contrib.summary.histogram(w.name, w)
        
        
    if epoch > 200:
        #d_learning_rate.assign(0.0001 + random.random() * 0.0005)
        #g_learning_rate.assign(0.0001 + random.random() * 0.0005)
        d_learning_rate.assign(d_learning_base_rate * (0.9998 ** (epoch-200)))
        g_learning_rate.assign(g_learning_base_rate * (0.9998 ** (epoch-200)))
        print("d_learning_rate = {}, g_learning_rate = {}".format(d_learning_rate.numpy(), g_learning_rate.numpy()))

    #gaussian_rate.assign(base_gaussian_noise * (0.997 ** epoch))
        
    sample(epoch)
    if epoch % 4 == 0:
        tag = tags[int(epoch/4) % len(tags)]
        gen.save(GEN_WEIGHTS.format(tag))
        discrim.save(DISCRIM_WEIGHTS.format(tag))
    
