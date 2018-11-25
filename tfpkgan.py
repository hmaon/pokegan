import numpy as np
import queue
import threading
import time
import math


# TODO:
# * implement tf.train.Checkpoint
# * float16 ops
# * implement stabilization from Roth et al. (2017) https://arxiv.org/abs/1705.09367

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, GaussianNoise
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, MaxPool2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, ReLU, PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Softmax, Input, Concatenate, Add
from tensorflow.train import AdamOptimizer, RMSPropOptimizer, ProximalAdagradOptimizer
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
tf.app.flags.DEFINE_integer('batch_size', 20,
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
channels_shape=(3,64,64) # not bothering to set this from global config right now
                            
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
d_learning_base_rate = 0.0002
g_learning_base_rate = 0.0005
#base_gaussian_noise = .1

#METRICS=[keras.metrics.categorical_accuracy]

#INIT=tf.initializers.orthogonal(dtype = dtype)
INIT='orthogonal'
PRELUINIT=keras.initializers.Constant(value=0.2)
#g_batchnorm_constraint = MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=[0])
g_batchnorm_constraint = None

GEN_WEIGHTS="gen-weights-{}.hdf5"
DISCRIM_WEIGHTS="discrim-weights-{}.hdf5"

#VIRTUAL_BATCH_SIZE=batch_size
RENORM=True
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

def rangeshift(inp):
    inp = (inp * -1) + 127
    #print(inp)
    return inp

    
ImageDatagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=7,
        height_shift_range=7,
        zoom_range=0.1,
        fill_mode='wrap',
        cval=255,
        horizontal_flip=True,
        data_format='channels_first',
        dtype=dtype)

train_gen = ImageDatagen.flow_from_directory(
        FLAGS.data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='sparse',
        interpolation='lanczos')

num_classes = train_gen.num_classes + 1 # classes + fake

flowQueue = queue.Queue(5)

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
    out = np.uint8(out * 255)
    return out
    
def img_uint8_to_float(img):
    img = dtype(img)
    img *= 1./255
    return img

def img_quantize(img): # img is a Tensor 
    g = img
    g = tf.math.multiply(tf.constant(255.), g)
    #g = tf.math.floor(g) # quantize
    g = tf.math.multiply(tf.constant(1./255), g)            
    return g

# save a grid of images to a file
def render(all_out, filenum=0):            
    pad = 3
    swatchdim = 64 # 64x64 is the output of the generator
    swatches = 6 # per side
    dim = (pad+swatchdim) * swatches
    img = PIL.Image.new("RGB", (dim, dim), "white")

    for i in range(min(swatches * swatches, len(all_out))):
        out = all_out[i]
        out = img_float_to_uint8(out)
        #print("pre-move shape ", out.shape)
        out = np.moveaxis(out, 0, -1) # switch from channels_first to channels_last
        #print("check this: ", out.shape)
        swatch = PIL.Image.fromarray(out)
        x = i % swatches
        y = math.floor(i/swatches)
        #print((x,y))
        img.paste(swatch, (x * (pad+swatchdim), y * (pad+swatchdim)))

    img.save('out%d.png' %(filenum,))
    #ima = np.array(img).flatten()
    #ima = ima[:batch_size*swatchdim*swatchdim*3]
    #print("ima.shape: {}".format(ima.shape))
    #with writer.as_default():
    #    with tf.contrib.summary.always_record_summaries():
    #        tf.contrib.summary.image("sample", ima.reshape((batch_size,swatchdim,swatchdim,3)))

#test rendering and loading
render(next_batch()[0], -99999)
    
###
### D I S C R I M I N A T O R
###

def d_block(dtensor, depth = 128, stride=1, maxpool=False, stridedPadding='same'):

    # feature detection
    dtensor = Conv2D(depth, 3, strides=1,\
                              padding='same',\
                              kernel_initializer=INIT)(dtensor)
    dtensor = LeakyReLU(0.2)(dtensor)
    dtensor = BatchNormalization(axis=1, renorm=RENORM, )(dtensor)
    
    # strided higher level feature detection
    dtensor = Conv2D(depth, 3, strides=stride,\
                              padding=stridedPadding,\
                              kernel_initializer=INIT)(dtensor)
    dtensor = LeakyReLU(0.2)(dtensor)
    dtensor = BatchNormalization(axis=1, renorm=RENORM, )(dtensor)
    if maxpool:
        dtensor = MaxPool2D(padding='same')(dtensor)
        
    # nonsense?
    #dtensor = Conv2D(depth, 1, kernel_initializer=INIT)(dtensor)
    #dtensor = LeakyReLU(0.2)(dtensor)
    #dtensor = BatchNormalization(axis=1, renorm=RENORM, )(dtensor)    
    return dtensor

# unused
def res_d_block(dtensor, depth, stride = 1, stridedPadding='same'):

    if stride > 1:
        dtensor = Conv2D(depth, 3, strides=stride,\
                                  padding=stridedPadding,\
                                  kernel_initializer=INIT)(dtensor)
        dtensor = LeakyReLU(0.2)(dtensor)
        dtensor = BatchNormalization(axis=1, renorm=RENORM, )(dtensor)

    short = dtensor
    
    dtensor = Conv2D(depth, 3, strides=1,\
                              padding='same',\
                              kernel_initializer=INIT)(dtensor)
    dtensor = LeakyReLU(0.2)(dtensor)
    dtensor = BatchNormalization(axis=1, renorm=RENORM, )(dtensor)

    dtensor = Conv2D(depth, 3, strides=1,\
                              padding='same',\
                              kernel_initializer=INIT)(dtensor)
    dtensor = LeakyReLU(0.2)(dtensor)
    dtensor = BatchNormalization(axis=1, renorm=RENORM, )(dtensor)

    # diagram at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 ... seems weird but who am I to argue
    #if stride == 2:
    #    short = MaxPooling2D()(short)
    short = LeakyReLU(0.2)(short)  
    short = BatchNormalization(axis=1, renorm=RENORM, )(short)    
    
    short = SeparableConv2D(depth, 1)(short) 
    #short = PReLU()(short)
    short = LeakyReLU(0.2)(short)  
    short = BatchNormalization(axis=1, renorm=RENORM, )(short)    
    
    short = SeparableConv2D(depth, 1)(short)
    short = LeakyReLU(0.2)(short)
    short = BatchNormalization(axis=1, renorm=RENORM, )(short)        
        
    return Add()([dtensor, short])


#gaussian_rate = tfe.Variable(d_learning_base_rate)    # can't include in Discriminator() because .save() crashes on it :( :( :( wtf :(

def Discriminator():
    dense_dropout = 0.1
    
    inp = Input(channels_shape)    
    
    d = LeakyReLU(0.2)(inp) # clip negative input values from generator, as they're also clipped in image sampling
    
    d = GaussianNoise(.15)(d)
        
    d = Conv2D(64, 7, padding='same', kernel_initializer=INIT, strides=1)(d) # 64x64
    BatchNormalization(axis=1, renorm=RENORM, )(d)
    LeakyReLU(0.2)(d)          

    d = Conv2D(64, 7, padding='same', kernel_initializer=INIT, strides=2)(d) # 32x32
    BatchNormalization(axis=1, renorm=RENORM, )(d)
    LeakyReLU(0.2)(d)          
    
    d = res_d_block(d, 64, 2) # 16x16
    d = res_d_block(d, 64, 1)     
    d = res_d_block(d, 64, 1)     
    d = res_d_block(d, 64, 1)     
    
    d = res_d_block(d, 128, 2) # 8x8
    d = res_d_block(d, 128, 1) #     
    d = res_d_block(d, 128, 1) #     
    d = res_d_block(d, 128, 1) #     
    d = res_d_block(d, 256, 2) # 4x4
    d = res_d_block(d, 256, 1) #     
    d = res_d_block(d, 256, 1) #     
    d = res_d_block(d, 256, 1) #     
    d = res_d_block(d, 256, 2) # 2x2
    d = res_d_block(d, 256, 1) # 
    d = res_d_block(d, 256, 1) # 
    d = res_d_block(d, 256, 1) # 
    
    #d = AveragePooling2D()(d) # 1x1 mofo because that's what resnet does, I guess
    
    d = Flatten()(d)
    
    d = Dense(512, kernel_initializer=INIT)(d)
    d = LeakyReLU(0.2)(d)
    d = BatchNormalization(renorm=RENORM, )(d)
    
    # classify ??    
    d = Dense(num_classes, kernel_initializer=INIT)(d)
    #d = Softmax()(d) # calculated in softmax_cross_entropy() but not in hinge_loss()?
        
    discrim = Model(inputs=inp, outputs=d)    

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
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [batch_size,-1]), axis=1, keep_dims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [batch_size,-1]), axis=1, keep_dims=True)

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

def g_block(gtensor, depth=32, stride=1, size=3, upsample=True, deconvolve=False):
    conv = gtensor
    if upsample: 
        conv = UpSampling2D(dtype=dtype)(conv)
    
    if deconvolve:
        conv = Conv2DTranspose(depth, size, padding='same', strides=stride, kernel_initializer=INIT, dtype=dtype)(conv)
        conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=[1,2])(conv)     
        conv = BatchNormalization(axis=1, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)    
        
    #conv = SeparableConv2D(depth, size, padding='same', kernel_initializer=INIT, dtype=dtype)(conv)
    #conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=[1,2])(conv)  
    #conv = BatchNormalization(axis=1, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)            
    

    #conv = SeparableConv2D(depth, 3, depth_multiplier=2, padding='same', kernel_initializer=INIT)(conv)
    conv = Conv2D(depth, size, padding='same', kernel_initializer=INIT, dtype=dtype)(conv)
    conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=[1,2])(conv)
    
    #conv = Add()([conv, h])
    
    conv = BatchNormalization(axis=1, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)        

    conv = Conv2D(depth, 1, padding='same', kernel_initializer=INIT)(conv)
    conv = PReLU(alpha_initializer=PRELUINIT, shared_axes=[1,2])(conv)  
    conv = BatchNormalization(axis=1, renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(conv)            
    
    return conv
    
NOISE = 100
    
def Generator():
    input = g = Input((NOISE+num_classes,), dtype=dtype)

    zc = Reshape(target_shape=(1, 1, NOISE+num_classes))(input)
    #zc = UpSampling2D()(zc) # 2x2
    #zc = UpSampling2D()(zc) # 4x4
    #zc = UpSampling2D()(zc) # 8x8

    #g = Dense(512, kernel_initializer=INIT)(g)
    #g = PReLU(alpha_initializer=PRELUINIT)(g)
    #g = BatchNormalization(renorm=RENORM, )(g)

    g = Dense(1024, kernel_initializer=INIT, dtype=dtype)(g)
    g = PReLU(alpha_initializer=PRELUINIT)(g)
    g = BatchNormalization(renorm=RENORM, beta_constraint=g_batchnorm_constraint, gamma_constraint=g_batchnorm_constraint)(g)
    
    #g = Reshape(target_shape=(2,2,1024))(g)
    g = Reshape(target_shape=(1,1,1024))(g)

    g = g_block(g, 2048, 2)
    
    g = g_block(g, 1024, 2) # 4x4
    
    g = g_block(g, 512, 2) # 8x8

    #g = Concatenate()([g,zc])

    #zc = UpSampling2D()(zc) # 16x16
    
    g = g_block(g, 256, 2) # 16x16
    
    #zc = UpSampling2D()(zc) # 32x32
    
    g = g_block(g, 128, 2, size=5)  # 32x32
    
    #g = Concatenate()([g,zc])
    
    g = g_block(g, 64, 2, size=5) # 64x64

    # I don't know what these are supposed to do but whatever:
    #g = g_block(g, 64, 1, size=3, upsample=False, deconvolve=False) # 64x64

    #g = g_block(g, 64, 1, size=3, upsample=False, deconvolve=False) # 64x64

    g = Conv2D(64, 5, padding='same', kernel_initializer=INIT)(g)    
    #g = SeparableConv2D(256, 3, depth_multiplier=2, padding='same', kernel_initializer=INIT)(g)
    g = PReLU(alpha_initializer=PRELUINIT, shared_axes=[1,2])(g)
    g = BatchNormalization(axis=1, renorm=RENORM, )(g)

    #g = Conv2D(256, 1, padding='same', kernel_initializer=INIT)(g)    
    #g = SeparableConv2D(256, 3, depth_multiplier=2, padding='same', kernel_initializer=INIT)(g)    
    #g = SeparableConv2D(256, 3, depth_multiplier=2, padding='same', kernel_initializer=INIT)(g)
    #g = LeakyReLU(0.1)(g)
    #g = BatchNormalization(axis=1, renorm=RENORM, )(g)    
        
    #g = Conv2D(128, 1, padding='same', kernel_initializer=INIT)(g)
    #g = PReLU(alpha_initializer=PRELUINIT, shared_axes=[1,2])(g)
    #g = BatchNormalization(axis=1, renorm=RENORM, )(g)
    
    #print(g.shape)
    g = Conv2D(3, 1, activation='tanh', padding='same')(g)
    g = Reshape(channels_shape)(g) 
    
    gen = Model(inputs=input, outputs=g)
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
    noise = np.random.permutation(NOISE) * (1.0 / NOISE) % clipping
    if type(_class) == type(None):
        _class = keras.utils.to_categorical(random.randint(0, num_classes-2), num_classes=num_classes)
    return np.concatenate((_class, noise))

def gen_input_rand(clipping=1.0):
    return gen_input(clipping)

# optionally receives one-hot class array from training loop
def gen_input_batch(classes=None, clipping=.95):
    if type(classes) != type(None):
        return np.array([gen_input(cls, clipping) for cls in classes], dtype=npdtype)
    print("!!! Generating random batch in gen_input_batch()!")
    return np.array([gen_input_rand(clipping) for x in range(batch_size)], dtype=npdtype)

        
def sample(filenum=0):
    all_out = []
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 128, 129, 130, 131, 132, 133, 134, 135, 136, 119, 101, 93, 142, num_classes-1, 143]
    #print("[sample()] num classes: ", len(classes))
    _classidx=0
    for i in range(4):
        inp = gen_input_batch([keras.utils.to_categorical(classes[x] % num_classes, num_classes=num_classes) for x in range(_classidx,_classidx+10)], clipping=0.5)
        _classidx += 10
        print(inp.shape)
        batch_out = tf.nn.relu(gen.predict(inp)).numpy()
        print(batch_out.shape)
        for out in batch_out:
            all_out.append(out)
    render(all_out, filenum)
            
        

###
### optimizer config
###

g_learning_rate = tfe.Variable(g_learning_base_rate)
#gen_optimizer = RMSPropOptimizer(learning_rate=0.005)
gen_optimizer = AdamOptimizer(learning_rate=g_learning_rate, beta1=0.5)
#gen_optimizer = AdamWOptimizer(learning_rate=g_learning_rate, weight_decay=0.025)
    
d_learning_rate = tfe.Variable(d_learning_base_rate)
#discrim_optimizer = RMSPropOptimizer(learning_rate=0.01)    
discrim_optimizer = AdamOptimizer(learning_rate=d_learning_rate, beta1=0.5)
#discrim_optimizer = AdamWOptimizer(learning_rate=d_learning_rate, weight_decay = 0.025)

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
### T R A I N
###

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
    for batch_num in range(batches):
        start = timer()
        
        # get real data
        x,y = next_batch()
        x = tf.convert_to_tensor(x) 
        if len(y) != train_gen.batch_size:
            continue # avoid re-analysis of ops due to changing batch sizes; maybe was only relevant in plaidml? not sure
        
        y = real_y = np.array([keras.utils.to_categorical(cls, num_classes=num_classes) * 0.9 for cls in y], dtype=npdtype)
        all_fake_y = np.array([keras.utils.to_categorical(num_classes-1, num_classes = num_classes) * 0.9 for dummy in x], dtype=npdtype)
        
        weights = tf.ones_like(y) * 0.75 + real_y * (.25/.9) + all_fake_y * (.25/.9)
        
        for d_step in range(2):
            with tf.GradientTape() as dtape, tf.GradientTape() as gtape:
                #tape.watch(x)
                
                if d_step == 0:
                    real_out = discrim(x, training=True)
                    d_loss = tf.losses.softmax_cross_entropy(real_y, real_out)
                    #d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_out, labels=y)
                    #d_loss_real = tf.losses.hinge_loss(labels=y, logits=real_out, weights=weights)
                    #d_loss_real *= d_loss_real # square hinge
                    d_acc = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(real_y, real_out))                
                if d_step == 1:
                    half_real_y = real_y[:len(real_y) >> 1]
                    half_fake_y = all_fake_y[:len(half_real_y)]
                    x_gen_input = gen_input_batch(half_real_y) # real classes with appended random noise inputs
                    gen_x = gen(x_gen_input, training=True)

                    cumulative_x = tf.concat([x, gen_x[:len(half_fake_y)]], -4)
                                        
                    cumulative_RF_y = tf.concat([real_y, half_fake_y], -2)

                    out = discrim(cumulative_x, training=True)
                    
                    d_loss = tf.losses.softmax_cross_entropy(cumulative_RF_y, out)
                    d_acc_real = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(real_y, out[:len(real_y)]))
                    d_acc_fake = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(half_fake_y, out[len(real_y):]))
                    
                
                    # call regularizer from https://github.com/rothk/Stabilizing_GANs
                    #disc_reg = Discriminator_Regularizer(real_out, x, gen_out, gen_x, tape)
                    #disc_reg = (gamma/2.0)*disc_reg
                    #d_loss += disc_reg

                    gen_out_a = out[len(real_y):]
                    
                    x_gen_input = gen_input_batch(real_y) # same classes, full spread, different noise
                    gen_x = gen(x_gen_input, training=True)
                    gen_out_b = discrim(gen_x, training=True)
                    gen_out = tf.concat([gen_out_a, gen_out_b], -2)
                    cumulative_real_y = tf.concat([half_real_y, real_y], axis=-2)

                    #print("shape check, gen_out_b ", gen_out_b.numpy().shape, "gen_out ", gen_out.numpy().shape, "cumulative_real_y ", cumulative_real_y.numpy().shape)
                    
                    g_loss = tf.losses.softmax_cross_entropy(cumulative_real_y, gen_out)
                    g_acc = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(cumulative_real_y, gen_out))

                
            gradients_of_discriminator = dtape.gradient(d_loss, discrim.variables)
            apply_discrim_gradients(gradients_of_discriminator, discrim)
            
            if d_step == 1: 
                gradients_of_generator = gtape.gradient(g_loss, gen.variables)
                apply_gen_gradients(gradients_of_generator, gen)
            else:
                real2_msg = "REAL2: {:.7f}, acc: {:.3f}".format(d_loss, d_acc/batch_size)
                
        end = timer()
        batches_timed += 1
        total_time += (end - start)
        
        #d_loss_real = tf.reduce_mean(d_loss_real)
        #d_loss_fake = tf.reduce_mean(d_loss_fake)
        with writer.as_default():
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("d_loss", d_loss)
                tf.contrib.summary.scalar("d_accuracy_real", d_acc_real/batch_size)
                #tf.contrib.summary.scalar("d_loss_fake", d_loss_fake)
                tf.contrib.summary.scalar("d_accuracy_fake", d_acc_fake/batch_size)
                tf.contrib.summary.scalar("g_loss", g_loss)
                tf.contrib.summary.scalar("g_accuracy", g_acc/batch_size)
                #tf.contrib.summary.histogram("g_grads?", gradients_of_generator[0]) 
        
        writer.flush()
        
        print("epoch {}; batch {} / {}".format(epoch, batch_num, batches))
        print("d_loss: {:.7f}".format(d_loss)) 
        print("real acc: {:.3f}".format(d_acc_real/batch_size))
        if real2_msg != None: print(real2_msg)
        print("fake acc: acc: {:.3f}".format(d_acc_fake/batch_size))
        #print("final d_loss: {}, regularizer delta: {}".format((d_loss), disc_reg))
        print("g_loss: {:.7f}, acc: {:.3f}".format((g_loss), g_acc/batch_size))
        #print("mean d grads: {:.8f}, mean g grads: {:.8f}".format(tf.reduce_mean(gradients_of_generator), tf.reduce_mean(gradients_of_discriminator)))
        print("batch time: {:.3f}, total_time: {:.3f}, mean time: {:.3f}\n".format(end-start, total_time, total_time / batches_timed))
        
        global_step.assign_add(1)
        
        
    if epoch > 150:
        d_learning_rate.assign(d_learning_base_rate * (0.997 ** (epoch-150)))
        g_learning_rate.assign(g_learning_base_rate * (0.997 ** (epoch-150)))
        print("d_learning_rate = {}, g_learning_rate = {}".format(d_learning_rate.numpy(), g_learning_rate.numpy()))

    #gaussian_rate.assign(base_gaussian_noise * (0.997 ** epoch))
        
    sample(epoch)
    if epoch % 4 == 0:
        tag = tags[int(epoch/4) % len(tags)]
        gen.save(GEN_WEIGHTS.format(tag))
        discrim.save(DISCRIM_WEIGHTS.format(tag))
    
