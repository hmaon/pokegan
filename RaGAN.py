import queue
import threading
import time
import math
import random,math
import sys,os
from timeit import default_timer as timer
from functools import partial



# TODO:
# * implement tf.train.Checkpoint
# * float16 ops XXX not functioning
# XXX implement stabilization from Roth et al. (2017) https://arxiv.org/abs/1705.09367 XXX
# * SELU!

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, GaussianNoise, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, MaxPool2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, ReLU, PReLU, ELU, AlphaDropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Softmax, Input, Concatenate, Add, Multiply
from tensorflow.train import AdamOptimizer, RMSPropOptimizer, ProximalAdagradOptimizer, GradientDescentOptimizer
from tensorflow.contrib.opt import NadamOptimizer, AdamWOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.constraints import MinMaxNorm

from keras_contrib.layers.normalization import InstanceNormalization

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras

import tensorflow.contrib.eager as tfe

import PIL
import h5py

from scipy import ndimage

import stddev

inorm_counter = 0
def InstanceNorm():
    # XXX broken - stops gradients, cause unknown
    #global inorm_counter
    #inorm_counter = inorm_counter+1
    #return Lambda(lambda x : x)
    return Lambda(tf.contrib.layers.instance_norm) #, name='InstanceNorm_%d' % (inorm_counter,))

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 56,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'dir_per_class',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16? [XXX NOT WORKING]""")
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.85,
                            """Fraction of GPU memory to reserve; 1.0 may crash the OS window manager.""")
tf.app.flags.DEFINE_integer('epoch', 1,
                            """Epoch number of last checkpoint for continuation! (e.g., number of last screenshot)""")
tf.app.flags.DEFINE_string('gen', None,
                           """Generator weights to load.""")
tf.app.flags.DEFINE_string('disc', None,
                           """Discriminator weights to load.""")
tf.app.flags.DEFINE_string('loss', 'ralsgan',
                           """Loss function: rel | raverage | ralsgan | rahinge [broken?] | plain""")
tf.app.flags.DEFINE_boolean('add_noise', False,
                            """Add gaussian noise to discriminator input?""")                           
                           
batch_size = FLAGS.batch_size
                           
dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
npdtype = np.float16 if FLAGS.use_fp16 else np.float32

data_format=tf.keras.backend.image_data_format()
channels_shape = (3,128,128) if data_format == 'channels_first' else (128,128,3)
print("channels_shape == ", channels_shape)
                            
# not sure whether this session stuff matters in eager execution mode :/
from tensorflow.keras.backend import set_session
config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
session = tf.Session(config=config)
set_session(session)

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

ctx = None

save_path = 'summary/'

global_step = tf.train.create_global_step()
writer = tf.contrib.summary.create_file_writer(save_path)
        
#keras.backend.set_floatx('float16')
    
EPOCHS = 10000
d_learning_base_rate = 0.001
g_learning_base_rate = 0.001
weight_decay = 1e-4

#METRICS=[keras.metrics.categorical_accuracy]

#INIT=tf.initializers.orthogonal(dtype = dtype)
INIT=tf.initializers.glorot_normal(dtype = dtype)
#INIT=tf.initializers.lecun_normal()
PRELUINIT=keras.initializers.Constant(value=0.0)
#g_batchnorm_constraint = MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=[0])
g_batchnorm_constraint = None

K_REG=tf.keras.regularizers.l2(weight_decay)
#K_REG=None
BNSCALE=True
PRELU_SHARED_AXES=[1,2]

GEN_WEIGHTS="gen-weights-{}.h5"
DISCRIM_WEIGHTS="discrim-weights-{}.h5"

#VIRTUAL_BATCH_SIZE=batch_size
RENORM=False
BATCH2D_AXIS = 1 if data_format == 'channels_first' else -1
CONCAT_AXIS = BATCH2D_AXIS
#INSTNORM_AXIS = 1 if data_format == 'channels_first' else -1
INSTNORM_AXIS = None

load_disc=None
load_gen=None
start_epoch=FLAGS.epoch

tags = ["A", "B", "C", "D"]

if start_epoch > 1:
    tag = tags[ math.floor(start_epoch/10) % len(tags) ]
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


### hack from https://github.com/farizrahman4u/seq2seq/issues/129#issuecomment-260949294
### not used since fixing normalization.py from keras_contrib
def hack_save(model, fileName):
    file = h5py.File(fileName,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight'+str(i),data=weight[i])
    file.close()

def hack_load(model, fileName):
    file=h5py.File(fileName,'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    
###
### D A T A S E T
###


def get_input_noise_scale(epoch):
    scale = 0.2 * (0.999 ** (epoch + 100)) # effect of this is delayed due to queue but oh well; the constant here (100) is the length of the queue
    scale = max(scale, 2.5/256)
    return scale

input_noise_scale = get_input_noise_scale(start_epoch)

def prepro(inp):
    if FLAGS.add_noise:
        inp += np.random.normal(scale = input_noise_scale * 127., size=inp.shape)
    inp = inp - 127.
    inp = np.clip(inp, -127., 127.)
    return inp

with tf.device('/cpu:0'): 
    ImageDatagen = ImageDataGenerator(
            rescale=1./127,
            width_shift_range=10,
            height_shift_range=10,
            zoom_range=0.,
            fill_mode='reflect',
            cval=255,
            horizontal_flip=False,
            data_format=data_format,
            preprocessing_function=prepro,
            dtype=dtype)

    train_gen = ImageDatagen.flow_from_directory(
            FLAGS.data_dir,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='sparse',
            interpolation='lanczos')

    num_classes = train_gen.num_classes

    flowQueue = queue.Queue(100)

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
    def render(all_out, fileidx="0-0"):            
        pad = 3
        swatchdim = 128 # 64x64 is the output of the generator
        swatches = 8 # per side
        dim = (pad+swatchdim) * swatches
        img = PIL.Image.new("RGB", (dim, dim), "white")

        for i in range(min(swatches * swatches, len(all_out))):
            out = all_out[i]
            out = out + 1.0
            #out = tf.clip_by_value(out, 0., 2.).numpy()
            out = np.clip(out, 0., 2.)
            out = img_float_to_uint8(out)
            #print("pre-move shape ", out.shape)
            if data_format == 'channels_first':
                out = np.moveaxis(out, 0, -1) # switch from channels_first to channels_last
            #print("check this: ", out.shape)
            swatch = PIL.Image.fromarray(out)
            x = i % swatches
            y = math.floor(i/swatches)
            #print((x,y))
            img.paste(swatch, (x * (pad+swatchdim), y * (pad+swatchdim)))

        img.save('out%s.jpeg' %(fileidx,))
        #ima = np.array(img).flatten()
        #ima = ima[:batch_size*swatchdim*swatchdim*3]
        #print("ima.shape: {}".format(ima.shape))
        #with writer.as_default():
        #    with tf.contrib.summary.always_record_summaries():
        #        tf.contrib.summary.image("sample", ima.reshape((batch_size,swatchdim,swatchdim,3)))

#test rendering and loading
render(next_batch()[0], "-test")
print("sample render done")

   
###
### D I S C R I M I N A T O R
###

def simple_d_block(dtensor, depth = 128, stride=1, stridedPadding='same', size=5):
    dtensor = Conv2D(depth, size, strides=stride,\
                              padding=stridedPadding, use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    dtensor = LeakyReLU(alpha=0.2)(dtensor)
    dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
    #dtensor = InstanceNormalization(axis=INSTNORM_AXIS)(dtensor)
        
    return dtensor

    
def separable_d_block(dtensor, depth = 128, stride=1, stridedPadding='same', size=3):
    dtensor = SeparableConv2D(depth, size, strides=stride,\
                              padding=stridedPadding, depth_multiplier=2, use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    dtensor = LeakyReLU(alpha=0.2)(dtensor)
    dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
    #dtensor = InstanceNormalization(axis=INSTNORM_AXIS)(dtensor)
    
    return dtensor
    
    
def halfpool_d_block(dtensor, depth = 128, stride=2, stridedPadding='same', size=5):
    
    pool = AveragePooling2D()(dtensor)
    
    dtensor = Conv2D(depth, size, strides=stride,\
                              padding=stridedPadding, use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    dtensor = LeakyReLU(alpha=0.2)(dtensor)
    
    dtensor = Concatenate(axis=CONCAT_AXIS)([dtensor, pool])
    
    dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
        
    # nonsense? mix back down to depth, not depth*2
    dtensor = Conv2D(depth, 1, kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    dtensor = LeakyReLU(alpha=0.2)(dtensor)
    dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)    
    
    return dtensor
    
    
# ResNet-like implementation
def res_d_block(dtensor, depth, stride = 1, stridedPadding='same', extra=3):

    if stride > 1:
        dtensor = Conv2D(depth, 3, strides=stride,\
                                  padding=stridedPadding,\
                                  kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(dtensor)
        dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
        dtensor = LeakyReLU(alpha=0.2)(dtensor)

    short = dtensor

    for i in range(extra):
        dtensor = Conv2D(depth, 3, strides=1,\
                                  padding='same',\
                                  kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False)(dtensor)
        dtensor = LeakyReLU(alpha=0.2)(dtensor)
        dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)
    
    # diagram at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 ... re: full pre-activation, sort of
    for i in range(2):
        short = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(short)    
        short = LeakyReLU(alpha=0.2)(short)
            
        short = Conv2D(depth, 1, strides=1,\
                      padding='same',\
                      kernel_initializer='orthogonal', kernel_regularizer=K_REG, use_bias=False)(short)
            
    dtensor = Add()([dtensor, short])
    return dtensor

# semi-DenseNet-like block    
def dense_d_block(dtensor, depth = 128, stride=1, stridedPadding='same', size=5, extra=2):

    dtensor = Conv2D(depth, size, strides=stride,\
                              padding=stridedPadding, use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)
    
    for i in range(extra):
        dt2 = LeakyReLU(alpha=0.2)(dtensor)
        dt2 = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dt2)            
        
        dt2 = Conv2D(depth, size, strides=1,\
                                  padding=stridedPadding, use_bias=False,\
                                  kernel_initializer=INIT, kernel_regularizer=K_REG)(dt2)

        dtensor = Concatenate(axis=CONCAT_AXIS)([dtensor, dt2])            

    dtensor = LeakyReLU(alpha=0.2)(dtensor)
    dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)            

    dtensor = Conv2D(depth, 1, strides=stride,\
                              padding=stridedPadding, use_bias=False,\
                              kernel_initializer=INIT, kernel_regularizer=K_REG)(dtensor)

    dtensor = LeakyReLU(alpha=0.2)(dtensor)
    dtensor = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(dtensor)            
        
    return dtensor

    

#gaussian_rate = tfe.Variable(d_learning_base_rate)    # can't include in Discriminator() because .save() crashes on it :( :( :( wtf :(

def Discriminator():
    dense_dropout = 0.25
    
    inp = Input(channels_shape)    
    #print("discrim input shape:", inp.shape)
    
    d = inp
    
    d = simple_d_block(d, 16, stride=1, size=5) # 128x128
    
    d = simple_d_block(d, 32, 2) # 64x64
    
    d = simple_d_block(d, 64, 2) # 32x32

    d = simple_d_block(d, 128, 2) # 16x16

    #d = LeakyReLU(alpha=0.2)(d)
    #d = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, )(d)
    
    d = simple_d_block(d, 256, 2) # 8x8

    d = simple_d_block(d, 512, 2) # 4x4
    
    d = simple_d_block(d, 1024, size=3, stride=2) # 2x2
    
    #d = simple_d_block(d, 1024, 2, size=2) # 1x1


    d = Flatten()(d)
    
    #d = Concatenate()([d, a])

    #e = d
    
    d = Dense(1024, kernel_initializer='lecun_normal', kernel_regularizer=K_REG, use_bias=False, name='dense_filler1')(d)
    #e = LeakyReLU(alpha=0.2)(e)
    d = Activation('selu')(d)
    intermed = AlphaDropout(dense_dropout)(d)
    #e = BatchNormalization(trainable=True, renorm=RENORM)(e) # turning this on seems to break the discriminator and it's not clear why

    # classify ??    
    d = Dense(num_classes, kernel_initializer='lecun_normal', kernel_regularizer=K_REG, use_bias=False, name='dense_classifer')(intermed) # classifier    

    e = Concatenate()([d, intermed])
    e = Activation('selu')(e)
    e = Dense(1, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, name='dense_discrim')(e) # realness
        
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
# this needed to be converted to work with eager execution and also multiple image classes; currently broken
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
def dense_g_block(gtensor, depth=32, stride=2, size=3, upsample=True, deconvolve=False, extra=1, extrasize=3, final_down_conv=True):
    assert extra > 0
    conv = gtensor
    if upsample: 
        conv = UpSampling2D(dtype=dtype)(conv)
        conv = Conv2D(depth, size, padding='same', kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(conv)
    
    if deconvolve:
        conv = Conv2DTranspose(depth, size, padding='same', strides=stride, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(conv)
            
    if extra > 0:
        maps = [conv] # closer to true DenseNet architecture
        #maps = [] # lighter-weight version forcing all deconv output through conv2ds
                      
        for i in range(extra):
            if len(maps) > 1:         
                conv = Concatenate(axis=CONCAT_AXIS)(maps)

            conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=PRELU_SHARED_AXES)(conv)    
            conv = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)        
                
            conv = Conv2D(depth, extrasize, padding='same', kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(conv)
            maps.append(conv)

        conv = Concatenate(axis=CONCAT_AXIS)(maps)
        conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=PRELU_SHARED_AXES)(conv)    
        conv = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)        
        
        if final_down_conv:
            conv = Conv2D(depth, 1, padding='same', kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(conv)
            conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=PRELU_SHARED_AXES)(conv)    
            conv = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)        
            
    return conv


def simple_g_block(gtensor, depth=32, stride=1, size=3, upsample=True, deconvolve=False, extra=0, padding='same', input_dim=None):
    conv = gtensor
    if upsample: 
        if stride >= 2:
            conv = UpSampling2D(dtype=dtype)(conv) # needs tf-nightly, doesn't work in 1.12
        if stride == 4:
            conv = UpSampling2D(dtype=dtype)(conv) 
        conv = Conv2D(depth, size, padding='same', kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(conv)
    
    if deconvolve:
        conv = Conv2DTranspose(depth, size, padding='same', strides=stride, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype, )(conv)

    #conv = PReLU(alpha_initializer=PRELUINIT)(conv) 
    conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=PRELU_SHARED_AXES)(conv)    
    #conv = ELU()(conv)
    conv = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, scale=BNSCALE, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)        
    #conv = InstanceNormalization(axis=INSTNORM_AXIS, scale=False)(conv)
            
    return conv

    
    
NOISE = 100
NOIS2 = 2
    
def Generator():
    xz = Input((num_classes * NOIS2 + NOISE,), dtype=dtype)    
    
    g = xz
    

    
    g = Dense(1024, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(g)    
    g = PReLU(alpha_initializer=PRELUINIT)(g)
    g = AlphaDropout(0.25)(g)
    #g = BatchNormalization(trainable=True, renorm=RENORM, scale=True, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(g)
    
    g = Dense(4096, kernel_initializer=INIT, kernel_regularizer=K_REG, use_bias=False, dtype=dtype)(g)    
    g = PReLU(alpha_initializer=PRELUINIT)(g)
    #g = AlphaDropout(0.25)(g)
    #g = BatchNormalization(trainable=True, renorm=RENORM, scale=True, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(g)
    
    g = Reshape(target_shape=(1024,2,2) if data_format == 'channels_first' else (2,2,1024))(g)
    #g = BatchNormalization(trainable=True, axis=BATCH2D_AXIS, renorm=RENORM, scale=BNSCALE, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(g)
    #g = InstanceNormalization(axis=INSTNORM_AXIS)(g)
    
    #g = simple_g_block(g, 512, size=4, stride=4, padding='valid', upsample=False, deconvolve=True)

    g = simple_g_block(g, 512, stride=2) # 4x4

    print("4x4 shape? ", g.shape)
        
    g = simple_g_block(g, 256+128, stride=2) # 8x8

    g = simple_g_block(g, 128+64, stride=2) # 16x16    
    
    g = simple_g_block(g, 64+32, stride=2)  # 32x32
            
    g = simple_g_block(g, 32+16, stride=2) # 64x64
    
    g = simple_g_block(g, 24, stride=2) # 128x128
    
    print(g.shape)
    g = Conv2D(3, kernel_size=1, kernel_initializer=INIT, kernel_regularizer=K_REG, activation='tanh', padding='same', use_bias=False, dtype=dtype)(g)
    g = Reshape(channels_shape)(g) 
    #print("Generator output shape", g.shape)
    
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
    noise = np.random.normal(loc = 0, scale=1, size=NOISE)
    noise = (noise % clipping) * (np.abs(noise) / noise)
    #nois2 = np.random.normal(loc = 0, scale=1, size=NOIS2)
    #nois2 = (nois2 % clipping) * (np.abs(nois2) / nois2)
    #nois2[0] = .5
    #nois2[1] = -.5
    nois2 = np.array([.5,-.5])
    if type(_class) == type(None):
        _class = keras.utils.to_categorical(random.randint(0, num_classes-2), num_classes=num_classes)
    ret = np.multiply(_class.reshape((-1,1)), nois2).flatten()
    ret = np.concatenate( (ret, noise) )
    #print("input shape: ", ret.shape)
    return ret

def gen_input_rand(clipping=1.0):
    return gen_input(clipping)

# optionally receives one-hot class array from training loop
def gen_input_batch(classes=None, clipping=1):
    if type(classes) != type(None):
        return np.array([gen_input(cls, clipping) for cls in classes], dtype=npdtype)
    print("!!! Generating random batch in gen_input_batch()!")
    return tf.convert_to_tensor(np.array([gen_input_rand(clipping) for x in range(batch_size)], dtype=npdtype))

        
def sample(fileidx="0-0"):
    all_out = []
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49] #, 128, 129, 130, 131, 132, 133, 134, 135, 136, 119, 101, 93, 142, num_classes-1, 143]
    #print("[sample()] num classes: ", len(classes))
    _classidx=0
    for i in range(4):
        class_one_hot = [keras.utils.to_categorical(classes[x % len(classes)] % num_classes, num_classes=num_classes) for x in range(_classidx,_classidx+10)]
        class_one_hot = class_one_hot +  [(keras.utils.to_categorical(classes[(x + _classidx) % len(classes)] % num_classes, num_classes=num_classes) * ((x+1) / 10.) + keras.utils.to_categorical((x + 1 + _classidx) % num_classes, num_classes=num_classes) * ((10 - x) / 10. ))  for x in range(0,10)] 
        inp = gen_input_batch(class_one_hot, clipping=0.5*(i+2))
        _classidx += 20
        print(inp.shape)
        batch_out = gen.predict(inp)
        print(batch_out.shape)
        for out in batch_out:
            all_out.append(out)
    render(all_out, fileidx)
        

###
### optimizer config
###

g_learning_rate = tfe.Variable(g_learning_base_rate)
g_weight_decay = tfe.Variable(0.0001) # continuously reset in training
#gen_optimizer = RMSPropOptimizer(learning_rate=g_learning_rate)
#gen_optimizer = AdamOptimizer(learning_rate=g_learning_rate, beta1=0.5)
gen_optimizer = AdamWOptimizer(learning_rate=g_learning_rate, beta1=0.5, weight_decay = g_weight_decay)
    
d_learning_rate = tfe.Variable(d_learning_base_rate)
d_weight_decay = tfe.Variable(0.0001)
#discrim_optimizer = RMSPropOptimizer(learning_rate=d_learning_rate)    
#discrim_optimizer = AdamOptimizer(learning_rate=d_learning_rate, beta1=0.5)
discrim_optimizer = AdamWOptimizer(learning_rate=d_learning_rate, beta1=0.5, weight_decay = d_weight_decay)
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
sample("0-0")
#train_generator.reset()
#exit()

#print(gen.to_json())
#gen.save_weights("test.h5")

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

step_count = 0
  
for epoch in range(start_epoch,EPOCHS+1):
    print("--- epoch %d ---" % (epoch,))
    input_noise_scale = get_input_noise_scale(epoch)
    for batch_num in range(batches):
        step_count = step_count + 1
        start = timer()
        
        # get real data
        x,y = next_batch()
        if len(y) != train_gen.batch_size:
            continue # avoid various problems from changes in batch size
        x = tf.convert_to_tensor(x) 
        
        y = real_y = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) for cls in y], dtype=npdtype)
        yG  = np.array([keras.utils.to_categorical(random.randint(0, num_classes-1), num_classes=num_classes) for x in range(batch_size)], dtype=npdtype) # random samples for fake output
        yG2 = np.array([keras.utils.to_categorical(random.randint(0, num_classes-1), num_classes=num_classes) for x in range(batch_size)], dtype=npdtype) # random samples for fake output        
        
        #yG=tf.convert_to_tensor(yG)
        #yG2=tf.convert_to_tensor(yG2)
        
        ### non-class version:
        #yG = tf.zeros_like(yG)
        #yG2 = tf.zeros_like(yG2)
        
        #weights = tf.ones_like(y) * 0.75 + real_y * (.25/.9) + all_fake_y * (.25/.9)
        #weights = real_y / .9 # applied only to generator; concentrate only on maximizing similarity to real output, not on beating discriminator at fake vs. real game
        
        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            #dtape.watch(x)
            #gtape.watch(x)
            
            x_gen_input = gen_input_batch(y) 
            gen_x = gen(x_gen_input, training=True)
            
                        
            #if batch_num % 3 == 0:
            #    replayQueue.insert(0, gen_x.numpy())
        
            if FLAGS.add_noise:
                gen_x += np.random.normal(scale=input_noise_scale, size=gen_x.shape)
            #print("gen_x shape", gen_x.shape)
            #print("    x shape", x.shape)

            d_class,dx_realness = discrim(x, training=True)
            g_class,df_realness = discrim(gen_x, training=True)               

            
                        
            d_loss_real = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(dx_realness), logits = dx_realness)  # human-readable instrumentation
            d_loss_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(df_realness), logits = df_realness)
            
            if FLAGS.loss == 'plain':
                d_loss = d_loss_real + d_loss_fake
            elif FLAGS.loss == 'raverage':
                d_loss_A = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness), logits = dx_realness - tf.math.reduce_mean(df_realness)) 
                d_loss_B = tf.losses.sigmoid_cross_entropy(tf.zeros_like(df_realness), logits = df_realness - tf.math.reduce_mean(dx_realness))
                            
                d_loss = d_loss_A + d_loss_B
                d_loss *= .5
            elif FLAGS.loss == 'ralsgan':
                # (y_hat-1)^2 + (y_hat+1)^2
                d_loss = (tf.reduce_mean((dx_realness - tf.reduce_mean(df_realness) - tf.ones_like(dx_realness)) ** 2) + tf.reduce_mean((df_realness - tf.reduce_mean(dx_realness) + tf.ones_like(dx_realness)) ** 2))/2            
            elif FLAGS.loss == 'rel':                
                d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness) * 1.0, logits = dx_realness - df_realness) # simplest relativistic
            elif FLAGS.loss == 'rahinge':
                d_loss_A = tf.reduce_mean(tf.nn.relu(1.0 - (dx_realness - tf.reduce_mean(df_realness))))
                d_loss_B = tf.reduce_mean(tf.nn.relu(1.0 + (df_realness - tf.reduce_mean(dx_realness))))
                
                d_loss = d_loss_A + d_loss_B
                d_loss *= .5
                
            #d_loss_class = -1
            #d_loss_class = tf.losses.sigmoid_cross_entropy(y, logits = d_class - g_class) # idfk if this works
            #d_loss_class = tf.losses.sigmoid_cross_entropy(y, logits = d_class) + tf.losses.sigmoid_cross_entropy(tf.zeros_like(y), logits = g_class)
            #d_loss_class = tf.losses.softmax_cross_entropy(y * .9, logits = d_class) # leads to extreme collapse, all logits going negative, weird
            d_loss_class = tf.losses.hinge_loss(labels = y, logits = d_class) + tf.losses.hinge_loss(labels = tf.zeros_like(y), logits = g_class)
            
            d_loss = d_loss_class * 0.5 + d_loss * 0.5

            if len(replayQueue) > 60 and batch_num % 1 == 0:
                random.shuffle(replayQueue)
                gen_r = replayQueue.pop()
                _,dreplay_realness = discrim(gen_r, training=True)                           
                #df_realness = dreplay_realness
                # should factor out loss functions into separate function...
                d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness), dx_realness - dreplay_realness)
                #d_loss *= .5
                #d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness), dx_realness - dreplay_realness)            

            

            d_l2 = tf.add_n(discrim.losses)
            #d_l2 = -1
            #d_loss += d_l2 * .01

            
            # call regularizer from https://github.com/rothk/Stabilizing_GANs
            #disc_reg = Discriminator_Regularizer(real_out, x, gen_out, gen_x, tape)
            #disc_reg = (gamma/2.0)*disc_reg
            #d_loss += disc_reg

            #df_realness = tf.concat([df_realness[8:], df_realness[:8]], axis=0)
            
            #gen_out_a = out[len(real_y):]
        
            x_gen_input = gen_input_batch(y)  
            gen_x = gen(x_gen_input, training=True)
            gen_x = tf.convert_to_tensor(gen_x)
            g_class,g_realness = discrim(gen_x, training=True)

            if FLAGS.loss == 'plain':
                g_loss =  tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(df_realness) * .9, logits = df_realness)
            elif FLAGS.loss == 'raverage':
                g_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dx_realness), logits = dx_realness - tf.math.reduce_mean(df_realness)) 
                + tf.losses.sigmoid_cross_entropy(tf.ones_like(df_realness), logits = df_realness - tf.math.reduce_mean(dx_realness)) 
                g_loss *=  .5
            elif FLAGS.loss == 'ralsgan':
                g_loss = (tf.reduce_mean((dx_realness - tf.reduce_mean(df_realness) + tf.ones_like(dx_realness)) ** 2) + tf.reduce_mean((df_realness - tf.reduce_mean(dx_realness) - tf.ones_like(dx_realness)) ** 2))/2
            elif FLAGS.loss == 'rel':
                g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dx_realness) * 1.0, logits = df_realness - dx_realness) # simplest relativistic
            elif FLAGS.loss == 'rahinge':
                g_loss_A = tf.reduce_mean(tf.nn.relu(1.0 + (dx_realness - tf.reduce_mean(df_realness))))
                g_loss_B = tf.reduce_mean(tf.nn.relu(1.0 - (df_realness - tf.reduce_mean(dx_realness))))                
                g_loss = g_loss_A + g_loss_B
                g_loss *= .5
            else:
                raise BaseException("typo in loss function name? " + FLAGS.loss)
                

            #g_loss_class = -1
            #g_loss_class = tf.losses.sigmoid_cross_entropy(y, logits = g_class - d_class) # don't know if this works either
            #g_loss_class = tf.losses.sigmoid_cross_entropy(y, logits = g_class) # don't know if this works either
            #g_loss_class = tf.losses.softmax_cross_entropy(y * .9, logits = g_class)
            g_loss_class = tf.losses.hinge_loss(labels = y, logits = g_class) # don't know if this works either
            
            g_loss_nonrel = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(df_realness), logits=df_realness) # for human-readable instrumentation

            g_loss = g_loss * .7 + g_loss_class * .3
            #g_loss = g_loss_class
            
            g_l2 = tf.add_n(gen.losses)
            #g_l2 = -1
            #g_loss += g_l2 * .01

            #if g_l2 > 10.:
            #    g_loss += g_l2 * .25
        
            #if d_l2 > 15.:
           #    d_loss += d_l2 * .25
                
            if g_l2 < 10:
                g_weight_decay.assign(0.000003)
            elif g_l2 > 13:
                g_weight_decay.assign(0.0004)
            
            if d_l2 < 10:
                d_weight_decay.assign(0.000003)
            elif d_l2 > 13:
                d_weight_decay.assign(0.0004)
                
            #end with

            # simplistic implementation of https://openreview.net/forum?id=ByxPYjC5KQ
            # C is calculated as a lerp between x and gen_x. E from x to latent space is not implemented and likely will never be implemented


            #y = tf.convert_to_tensor(y)
            #with tf.GradientTape() as gp_tape:
            #    gp_tape.watch(x)
            #    gp_tape.watch(gen_x)
            #    #gp_tape.watch(y)
            #    gp_class, gp_realness = discrim(x * .5 + gen_x * .5, training=True)
            #    #gp_intermediate_loss = tf.losses.softmax_cross_entropy(y * 9, gp_class) * .5 + tf.losses.sigmoid_cross_entropy(tf.ones_like(gp_realness), gp_realness)            
            #    gp_intermediate_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(gp_realness), gp_realness)            
            
            #gradients_gp = gp_tape.gradient(gp_intermediate_loss, discrim.variables)
            
            #while gradients_gp[-1] == None: gradients_gp.pop()
            #for i in range(len(gradients_gp)): gradients_gp[i] = tf.reshape(gradients_gp[i], (-1,))
            #gradients_gp = tf.concat(gradients_gp, axis=-1)
            #gradients_gp_mean = tf.reduce_mean(gradients_gp) # just out of curiosity
            #print(gradients_gp.shape)
            #gp_lambda = 0.0001 # the authors set lambda to 10 so that tells us that something's is very wrong with this implementation
            #gp_loss = tf.norm(gradients_gp, ord=2) * gp_lambda 
            #d_loss += gp_loss

        #print(discrim.variables)

        gradients_of_discriminator = dtape.gradient(d_loss, discrim.variables)      
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
        
        if step_count % 25 == 0:
            sample("{}-{}".format(epoch, step_count))
        #print("dx_realness... {}".format(dx_realness.numpy().flatten()))
        print("d_class\n{}".format(d_class.numpy().flatten()))
        #print("y\n{}".format(y.flatten()))
        #print("grad_d_stddevs\n{}", grad_d_stddevs)
        #print("grad_d_mean\n{}".format(grad_d_mean))
        #print("grad_d_std_dev_from_zero: {}\ngrad_std_devs_from_zero:\n{}".format(grad_d_std_dev_from_zero, grad_std_devs_from_zero))
        print("epoch {}; batch {} / {}".format(epoch, batch_num, batches))
        #print("d_loss: {:.7f}, d_l2: {:.7f}".format(d_loss, d_l2)) 
        print("d_loss: {:.7}, d_loss_real: {:.7f}, d_loss_fake: {:.7f}, d_loss_class: {:.7f}, d_l2: {:.7f}".format(d_loss, d_loss_real, d_loss_fake, d_loss_class, d_l2))
        #print("d_realness MSE: {:.3f}, d_class MSE: {:.3f}".format(d_acc_realness, d_acc_class))
        #print("g_loss: {:.7f}, acc: {:.3f}".format((g_loss), g_acc/batch_size))
        print("g_loss: {:.7f}, g_loss_nonrel {:.7f}, g_loss_class: {:.7f}, g_l2: {:.7f}".format(g_loss, g_loss_nonrel, g_loss_class, g_l2))
        #print("gp_intermediate_loss: {:.7f}, gp_loss: {:.7f}, gradients_gp_mean: {:.7f}".format(gp_intermediate_loss, gp_loss, gradients_gp_mean))
        #print("mean d grads: {:.8f}, mean g grads: {:.8f}".format(tf.reduce_mean(gradients_of_generator), tf.reduce_mean(gradients_of_discriminator)))
        print("batch time: {:.3f}, total_time: {:.3f}, mean time: {:.3f}\n".format(end-start, total_time, total_time / batches_timed))       
        
        global_step.assign_add(1)

    #with writer.as_default():
    #    with tf.contrib.summary.always_record_summaries():
    #        for w in discrim.weights: tf.contrib.summary.histogram(w.name, w)
        
        
    if epoch > 200:
        #d_learning_rate.assign(0.0001 + random.random() * 0.0005)
        #g_learning_rate.assign(0.0001 + random.random() * 0.0005)
        d_learning_rate.assign(d_learning_base_rate * (0.9999 ** (epoch-200)))
        g_learning_rate.assign(g_learning_base_rate * (0.9999 ** (epoch-200)))
        print("d_learning_rate = {}, g_learning_rate = {}".format(d_learning_rate.numpy(), g_learning_rate.numpy()))

    #gaussian_rate.assign(base_gaussian_noise * (0.997 ** epoch))
        
    if epoch % 10 == 0:
        tag = tags[int(epoch/10) % len(tags)]
        gen.save_weights(GEN_WEIGHTS.format(tag))
        discrim.save_weights(DISCRIM_WEIGHTS.format(tag))
    
