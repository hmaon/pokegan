import numpy as np
import time


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
from tensorflow.contrib.opt import NadamOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint

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
tf.app.flags.DEFINE_integer('batch_size', 19,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'dir_per_class',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16. [XXX NOT WORKING]""")
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.85,
                            """Fraction of GPU memory to reserve; 1.0 may crash the OS window manager.""")
tf.app.flags.DEFINE_integer('epoch', 1,
                            """Epoch number of last checkpoint for continuation! (e.g., number of last screenshot)""")

dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
npdtype = np.float16 if FLAGS.use_fp16 else np.float32
                            
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
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

#METRICS=[keras.metrics.categorical_accuracy]

INIT=tf.orthogonal_initializer(dtype = dtype)

GEN_WEIGHTS="gen-weights-{}.hdf5"
DISCRIM_WEIGHTS="discrim-weights-{}.hdf5"


load_disc=None
load_gen=None
start_epoch=FLAGS.epoch

tags = ["A", "B", "C", "D"]

if start_epoch > 1:
    tag = tags[ start_epoch % len(tags) ]
    load_disc = DISCRIM_WEIGHTS.format(tag)
    load_gen = GEN_WEIGHTS.format(tag)
    start_epoch += 1

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

batch_size = 23    
    
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=3,
        height_shift_range=3,
        shear_range=5,
        fill_mode='nearest',
        cval=255,
        horizontal_flip=False,
        data_format='channels_first',
        dtype=dtype)

train_generator = train_datagen.flow_from_directory(
        FLAGS.data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='sparse',
        interpolation='lanczos')

num_classes = train_generator.num_classes + 1 # pokemon + fake

###
### D I S C R I M I N A T O R
###

def d_block(dtensor, depth = 128, stride=1, maxpool=False, stridedPadding='valid'):

    # feature detection
    dtensor = Conv2D(depth, 3, strides=1,\
                              padding='same',\
                              kernel_initializer=INIT)(dtensor)
    dtensor = PReLU()(dtensor)
    dtensor = BatchNormalization()(dtensor)
    
    # strided higher level feature detection
    dtensor = Conv2D(depth, 3, strides=stride,\
                              padding=stridedPadding,\
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
    dense_dropout = 0.1
    
    inp = Input((3,64,64))    
    
    d = GaussianNoise(0.1)(inp)
    
    d = inp
    
    d = Conv2D(32, 3, padding='valid', input_shape=(3,64,64), kernel_initializer=INIT)(d) # 62x62
    BatchNormalization()(d)
    PReLU()(d)
    
    d = Conv2D(64, 3, padding='valid', kernel_initializer=INIT, strides=2)(d) # 30x30
    BatchNormalization()(d)
    PReLU()(d)
       
       
    d = d_block(d, 256, 2) # 14x14
    d = d_block(d, 512, 2) # 6x6
    d = d_block(d, 512, 2) # 2x2
    #d = d_block(d, 512, 2) # 2x2
    
    d = Conv2D(512, 2, kernel_initializer=INIT, strides = 2)(d) 
    d = Flatten()(d)

    d = PReLU()(d)
    d = BatchNormalization()(d)
    
    # classify ??
    d = Dense(768, kernel_initializer=INIT)(d)
    d = PReLU()(d)
    d = BatchNormalization()(d)
    d = Dropout(dense_dropout)(d)
    
    #d = Dense(512, kernel_initializer=INIT)(d)
    #d = Dropout(dense_dropout)(d)
    #d = PReLU()(d)
    #d = BatchNormalization()(d)    
    
    d = Dense(num_classes, kernel_initializer=INIT)(d)
    #d = Softmax()(d) # calculated in softmax_cross_entropy()
        
    discrim = Model(inputs=inp, outputs=d)    

    return discrim

discrim = Discriminator()

discrim.summary()

if load_disc and os.path.isfile(load_disc):
    discrim.load_weights(load_disc)
else:
    print("not loading weights for discriminator")


discrim.call = tf.contrib.eager.defun(discrim.call)

d_learning_rate = tfe.Variable(0.0002)
#discrim_optimizer = RMSPropOptimizer(learning_rate=0.01)    
discrim_optimizer = NadamOptimizer(learning_rate=d_learning_rate)

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

def g_block(gtensor, depth=32, stride=1, size=3, upsample=True, deconvolve=True):
    conv = gtensor
    if upsample: 
        conv = UpSampling2D(dtype=dtype)(conv)
    
    if deconvolve:
        conv = Conv2DTranspose(depth, size, padding='same', strides=stride, kernel_initializer=INIT, dtype=dtype)(conv)
        conv = PReLU()(conv)     
        conv = BatchNormalization()(conv)    
        
    #conv = SeparableConv2D(depth, 3, depth_multiplier=2, padding='same', kernel_initializer=INIT)(conv)
    conv = Conv2D(depth, 3, padding='same', kernel_initializer=INIT, dtype=dtype)(conv)
    conv = PReLU()(conv)  
    conv = BatchNormalization()(conv)        

    #conv = Conv2D(depth, 1, padding='same', kernel_initializer=INIT)(conv)
    #conv = PReLU()(conv)  
    #conv = BatchNormalization()(conv)    
    
    return conv
    
NOISE = 50
    
def Generator():
    input = Input((NOISE+num_classes,), dtype=dtype)
    #g = Dense(512, kernel_initializer=INIT)(input)    
    #g = PReLU()(g)
    #g = BatchNormalization()(g)

    g = Dense(4*4*512, kernel_initializer=INIT, dtype=dtype)(input)
    g = PReLU()(g)
    g = BatchNormalization()(g)
    
    g = Reshape(target_shape=(512,4,4))(g)

    g = g_block(g, 1024, 2, size=2, upsample=False) # 8x8
    
    g = g_block(g, 512, 2, size=2, upsample=False) # 16x16
    
    g = g_block(g, 256, 2, size=2, upsample=False)  # 32x32
    
    g = g_block(g, 256, 2, size=2, upsample=False) # 64x64

    # I don't know what these are supposed to do but whatever:
    
    g = g_block(g, 128, 1, size=2, upsample=False, deconvolve=False) # 64x64
    
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

gen.call = tf.contrib.eager.defun(gen.call)

g_learning_rate = tfe.Variable(0.0005)
#gen_optimizer = RMSPropOptimizer(learning_rate=0.005)
gen_optimizer = NadamOptimizer(learning_rate=g_learning_rate)
    

# _class is one-hot category array
# randomized if None
def gen_input(_class=None, clipping=0.95):
    noise = np.random.uniform(0.0, 1.0, NOISE)
    if type(_class) == type(None):
        _class = keras.utils.to_categorical(random.randint(0, num_classes-2), num_classes=num_classes)
    return np.concatenate((_class, noise)) * clipping

def gen_input_rand(clipping=1.0):
    return gen_input(clipping)

# optionally receives one-hot class array from training loop
def gen_input_batch(classes=None, clipping=.95):
    if type(classes) != type(None):
        return np.array([gen_input(cls, clipping) for cls in classes], dtype=npdtype)
    print("!!! Generating random batch in gen_input_batch()!")
    return np.array([gen_input_rand(clipping) for x in range(batch_size)], dtype=npdtype)

def img_float_to_uint8(img, do_reshape=True):
    out = img
    if do_reshape: out = out.reshape(3, 64, 64)
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
    
def render(all_out, filenum=0):            
    pad = 3
    swatchdim = 64 # 64x64 is the output of the generator
    swatches = 5 # per side
    dim = (pad+swatchdim) * swatches
    img = PIL.Image.new("RGB", (dim, dim), "white")

    for i in range(min(swatches * swatches, len(all_out))):
        out = all_out[i]
        out = img_float_to_uint8(out)
        out = np.moveaxis(out, 0, -1) # switch from channels_first to channels_last
        #print("check this: ")
        #print(out.shape)
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
    
def sample(filenum=0):
    all_out = []
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 128, 129, 130, 131, 132, 133, 134, 135, 136, 119, 101, 93, 142, num_classes-1, 143]
    print(len(classes))
    _classidx=0
    for i in range(5):
        inp = gen_input_batch([keras.utils.to_categorical(classes[x] % num_classes, num_classes=num_classes) for x in range(_classidx,_classidx+5)], clipping=0.5)
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

#discrim.trainable = False

#adver = Sequential()
#adver.add(gen)
#adver.add(discrim)
#adver.summary()

#adver.compile(optimizer=RMSPropOptimizer(learning_rate=0.002), loss='kullback_leibler_divergence', metrics=METRICS) 

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

batches_timed = 0
total_time = 0

# for https://github.com/rothk/Stabilizing_GANs regularizer; 0.1 is a value they used without annealing in one of their examples 
gamma = 0.1

def apply_gradients(gradients_of_generator, gen, gradients_of_discriminator, discrim):
    gen_optimizer.apply_gradients(zip(gradients_of_generator, gen.variables))
    discrim_optimizer.apply_gradients(zip(gradients_of_discriminator, discrim.variables))

apply_gradients = tf.contrib.eager.defun(apply_gradients)    


for epoch in range(start_epoch,EPOCHS+1):
    print("--- epoch %d ---" % (epoch,))
    for batch_num in range(batches):
        start = timer()
        
        # get real data
        x,y = train_generator.next()
        x = tf.convert_to_tensor(x)
        if len(y) != train_generator.batch_size:
            continue # avoid re-analysis of ops due to changing batch sizes
        y = real_y = tf.convert_to_tensor(np.array([keras.utils.to_categorical(cls, num_classes=num_classes) * 0.9 for cls in y], dtype=npdtype))
        all_fake_y = tf.convert_to_tensor(np.array([keras.utils.to_categorical(num_classes-1, num_classes = num_classes) * 0.9 for dummy in x], dtype=npdtype))
        
        weights = tf.ones_like(y) * 0.5 + real_y * 0.5 + all_fake_y * 0.5
        
        #set_lr(discrim, disc_real_lr)
        
        with tf.GradientTape(persistent=True) as tape:
            #tape.watch(x)
            x_gen_input = gen_input_batch(real_y) # real classes with appended random noise inputs
            gen_x = gen(x_gen_input, training=True)

            real_out = discrim(x, training=True)
            #d_loss_real = tf.losses.softmax_cross_entropy(y, real_out)
            #d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_out, labels=y)
            d_loss_real = tf.losses.hinge_loss(labels=y, logits=real_out, weights=weights)
            #d_loss_real *= d_loss_real # square hinge
            d_acc_real = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(y, real_out))
            
            gen_out = discrim(gen_x, training=True)
            y = all_fake_y
            #d_loss_fake = tf.losses.softmax_cross_entropy(y, gen_out)
            #d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_out, labels=all_fake_y)
            d_loss_fake = tf.losses.hinge_loss(labels=all_fake_y, logits=gen_out, weights=weights)
            #d_loss_fake *= d_loss_fake # square hinge
            d_acc_fake = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(all_fake_y, gen_out))
            
            d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
            #d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
            #d_loss *= 0.5
            
            # call regularizer from https://github.com/rothk/Stabilizing_GANs
            #disc_reg = Discriminator_Regularizer(real_out, x, gen_out, gen_x, tape)
            #disc_reg = (gamma/2.0)*disc_reg
            #d_loss += disc_reg
            
            #g_loss = tf.losses.softmax_cross_entropy(real_y, gen_out)
            #g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_out, labels=real_y)
            g_loss = tf.losses.hinge_loss(labels=real_y, logits=gen_out, weights=weights)
            #g_loss *= g_loss # square hinge
            g_acc = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(real_y, gen_out))
            
            #x_gen_input = gen_input_batch(real_y) # same classes, different noise
            #gen_x = gen(x_gen_input, training=True)
            #gen_out = discrim(gen_x, training=True)
            #g_loss += tf.losses.softmax_cross_entropy(real_y, gen_out)
            #g_loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_out, labels=real_y)
            g_loss = tf.reduce_mean(g_loss)
            #g_loss *= 0.5

            
        gradients_of_generator = tape.gradient(g_loss, gen.variables)
        #gradients_of_generator = tf.gradients(g_loss, gen.variables)
        gradients_of_discriminator = tape.gradient(d_loss, discrim.variables)
        #gradients_of_discriminator = tf.gradients(d_loss, discrim.variables)        

        apply_gradients(gradients_of_generator, gen, gradients_of_discriminator, discrim)

        
        end = timer()
        batches_timed += 1
        total_time += (end - start)
        
        d_loss_real = tf.reduce_mean(d_loss_real)
        d_loss_fake = tf.reduce_mean(d_loss_fake)
        with writer.as_default():
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("d_loss_real", d_loss_real)
                tf.contrib.summary.scalar("d_accuracy_real", d_acc_real/batch_size)
                tf.contrib.summary.scalar("d_loss_fake", d_loss_fake)
                tf.contrib.summary.scalar("d_accuracy_fake", d_acc_fake/batch_size)
                tf.contrib.summary.scalar("g_loss", g_loss)
                tf.contrib.summary.scalar("g_accuracy", g_acc/batch_size)
                tf.contrib.summary.histogram("g_grads?", gradients_of_generator[0]) 
        
        writer.flush()
        
        print("REAL: {:.3f}, acc: {:.3f}".format(d_loss_real, d_acc_real/batch_size))
        print("FAKE: {:.3f}, acc: {:.3f}".format(d_loss_fake, d_acc_fake/batch_size))
        #print("final d_loss: {}, regularizer delta: {}".format((d_loss), disc_reg))
        print("G_LOSS: {:.3f}, acc: {:.3f}".format((g_loss), g_acc/batch_size))
        #print("mean d grads: {:.8f}, mean g grads: {:.8f}".format(tf.reduce_mean(gradients_of_generator), tf.reduce_mean(gradients_of_discriminator)))
        print("batch time: {:.3f}, total_time: {:.3f}, mean time: {:.3f}\n".format(end-start, total_time, total_time / batches_timed))
        
    if epoch > 100:
        d_learning_rate.assign(d_learning_rate * 0.997)
        g_learning_rate.assign(g_learning_rate * 0.997)
        print("d_learning_rate = {}, g_learning_rate = {}".format(d_learning_rate.numpy, g_learning_rate.numpy))
        
    sample(epoch)
    if epoch % 5 == 0:
        tag = tags[epoch % len(tags)]
        gen.save(GEN_WEIGHTS.format(tag))
        discrim.save(DISCRIM_WEIGHTS.format(tag))
    
