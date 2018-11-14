import numpy as np
import time

#import plaidml.keras
#plaidml.keras.install_backend()

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import HybridBlock, nn
import numpy as np

# TODO rewrite with gluon preprocessing
from keras.preprocessing.image import ImageDataGenerator

#import keras
import random,math
import sys,os
from timeit import default_timer as timer

import PIL

from scipy import ndimage

ctx = mx.gpu() # not sure...

    
EPOCHS = 5000

#METRICS=[keras.metrics.categorical_accuracy]

INIT='Xavier'

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
        batch_size=5,
        class_mode='sparse',
        interpolation='lanczos')

num_classes = train_generator.num_classes + 1 # number of image classes + fake

###
### D I S C R I M I N A T O R
###


class D_block(HybridBlock):
    def __init__(self, depth = 128, stride=1, maxpool=False, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.body = nn.HybridSequential()
            # feature detection
            self.body.add(nn.Conv2D(depth, 3, strides=1,\
                                      padding=1,\
                                      weight_initializer=INIT))
            self.body.add(nn.PReLU())
            self.body.add(nn.BatchNorm())
            
            # strided higher level feature detection
            self.body.add(nn.Conv2D(depth, 3, strides=stride,\
                                      padding=1,\
                                      weight_initializer=INIT))
            self.body.add(nn.PReLU())
            self.body.add(nn.BatchNorm())
            if maxpool:
                self.body.add(nn.MaxPool2D(3,2,1))
                
            # nonsense?
            self.body.add(nn.Conv2D(depth, 1, weight_initializer=INIT))
            self.body.add(nn.PReLU())
            self.body.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):
        #print(F)
        return self.body(x)

# unused
unused = """
def res_d_block(dtensor, depth, stride = 1):
    short = dtensor
    
    conv = d_block(dtensor, depth, stride)
    conv = d_block(conv, depth, 1)

    # diagram at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 ... seems weird but who am I to argue
    if stride == 2:
        short = AveragePooling2D()(short)
    short = PReLU()(short)  
    short = BatchNorm()(short)    
    short = SeparableConv2D(depth, 1)(short) 

    short = PReLU()(short)
    short = BatchNorm()(short)    
    short = SeparableConv2D(depth, 1)(short)
    short = PReLU()(short)
    short = BatchNorm()(short)    
    
        
    return Add()([conv, short])
"""
    
def Discriminator():
    dense_dropout = 0.05
    
    #inp = Input((3,64,64))
    
    d = nn.HybridSequential()
    
    # fine feature discrimation in two conv layers?
    d.add(nn.Conv2D(33, 5, padding=2, groups=3, weight_initializer=INIT))
    d.add(nn.BatchNorm())
    d.add(nn.PReLU())
    
    d.add(nn.Conv2D(66, 3, padding=1, groups=3, weight_initializer=INIT, strides=2))
    d.add(nn.BatchNorm())
    d.add(nn.PReLU())
    
    # 32x32 here
    d.add(D_block(256, 2)) # 16x16
    d.add(D_block(512, 2)) # 8x8
    d.add(D_block(512, 2)) # 4x4
    d.add(D_block(512, 2)) # 2x2
    
    #e = MaxPooling2D())
    #e = Flatten()(e)
    
    d.add(nn.Conv2D(512, 1, weight_initializer=INIT)) 
    d.add(nn.PReLU())
    d.add(nn.BatchNorm())
    
    d.add(nn.Flatten())
    #d.add(nn.Concatenate()([d,e])

    # classify ??
    d.add(nn.Dense(512, weight_initializer=INIT))
    d.add(nn.Dropout(dense_dropout))
    d.add(nn.PReLU())
    d.add(nn.BatchNorm())
    
    d.add(nn.Dense(512, weight_initializer=INIT))
    d.add(nn.Dropout(dense_dropout))
    d.add(nn.PReLU())
    d.add(nn.BatchNorm())    
    
    d.add(nn.Dense(num_classes, weight_initializer=INIT))
    #d.add(nn.Softmax())
        
    #discrim = Model(inputs=inp, outputs=d)    

    return d

discrim = Discriminator()

discrim.initialize(ctx=ctx)

discrim.summary(nd.zeros((train_generator.batch_size,3,64,64)))


if load_disc and os.path.isfile(load_disc):
    discrim.load_parameters(load_disc, ctx=ctx)
else:
    print("not loading weights for discriminator")

#discrim.hybridize()
trainerD = gluon.Trainer(discrim.collect_params(), 'RMSprop')

#exit()

###
### G E N E R A T O R
###

class G_block(HybridBlock):
    def __init__(self, depth=32, stride=1, size=3, upsample=True, **kwargs):
        super().__init__(**kwargs)
        self.upsample=upsample
        self.conv = nn.HybridSequential()
        with self.name_scope():
            #if upsample: 
            #    self.conv.add(nn.UpSampling2D())

            #conv.add(SeparableConv2D(depth, 3, depth_multiplier=2, padding='same', weight_initializer=INIT))
            self.conv.add(nn.Conv2D(depth, 3, padding=1, groups=1, weight_initializer=INIT))
            self.conv.add(nn.PReLU())  
            self.conv.add(nn.BatchNorm())
                
            #self.conv.add(self.conv2DTranspose(depth, size, padding='same', strides=stride, weight_initializer=INIT))
            #self.conv.add(PReLU())     
            #self.conv.add(BatchNorm())    

            self.conv.add(nn.Conv2D(depth, 1, padding=0, weight_initializer=INIT))
            self.conv.add(nn.PReLU())  
            self.conv.add(nn.BatchNorm())    
    
    def hybrid_forward(self, F, x):
        if self.upsample:
            # I think this is genuinely awful
            # the weights parameter is not needed with 'nearest' sample_type but then I still have to slice it off the bottom of the channels from the result; why is it even there?
            # also! the slicing isn't possible when x is of type Symbol!
            x = F.UpSampling(x, F.zeros_like(x), scale=2, sample_type='nearest', num_args=2)
        return self.conv(x)


class Reshape(HybridBlock):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape
    
    def hybrid_forward(self, F, x):
        return x.reshape(self.target_shape)
        
NOISE = 50
    
def Generator():
    g = nn.HybridSequential()
    
    g.add(nn.Dense(4*4*512, weight_initializer=INIT))
    g.add(nn.PReLU())
    g.add(nn.BatchNorm())
    
    g.add(Reshape(target_shape=(train_generator.batch_size,512,4,4)))

    g.add(G_block(512, 2, upsample=True)) # 8x8
    
    g.add(G_block(256, 2, upsample=True)) # 16x16
    
    g.add(G_block(128, 2, size=5, upsample=True))  # 32x32
    
    g.add(G_block(64, 2, size=5, upsample=True)) # 64x64

    # I don't know what these are supposed to do but whatever:
    
    g.add(G_block(64, 1, size=3, upsample=False)) # 64x64
    
    #g.add(SeparableConv2D(256, 3, depth_multiplier=2, padding='same', weight_initializer=INIT))
    #g.add(PReLU())
    #g.add(BatchNorm())

    #g.add(Conv2DTranspose(256, 3, padding='same', weight_initializer=INIT))
    #g.add(PReLU())
    #g.add(BatchNorm())
    
    #g.add(Conv2D(1024, 1, padding='same', weight_initializer=INIT))
    #g.add(PReLU())
    #g.add(BatchNorm())
    
    g.add(nn.Conv2D(3, 1, activation='sigmoid'))
    g.add(Reshape((train_generator.batch_size, 3, 64, 64))) # not sure if needed but we're doing channels_first; it helps as a sanity check when coding, at least!
    
    return g

gen = Generator()

gen.initialize(ctx=ctx)

gen.summary(nd.zeros((train_generator.batch_size,num_classes + NOISE)))

if load_gen and os.path.isfile(load_gen):
    gen.load_parameters(load_gen, ctx=ctx)
else:
    print("not loading weights for generator")

#gen.hybridize()
trainerG = gluon.Trainer(gen.collect_params(), 'RMSprop')

# _class is one-hot category array
# randomized if None
def gen_input(_class=None):
    noise = np.random.uniform(0.0, 1.0, NOISE)
    if type(_class) == type(None):
        _class = nd.one_hot(random.randint(0, num_classes-2), num_classes)[0].asnumpy() * 0.95
    return np.concatenate((_class, noise))

def gen_input_rand():
    return gen_input()

# optionally receives one-hot class array from training loop
def gen_input_batch(classes=None):
    if type(classes) != type(None):
        return nd.array(np.array([gen_input(cls) for cls in classes.asnumpy()]))
    print("!!! Generating random batch in gen_input_batch()!")
    return nd.array(np.array([gen_input_rand() for x in range(train_generator.batch_size)]))


def render(all_out, filenum=0):            
    pad = 3
    swatchdim = 64 # 64x64 is the output of the generator
    swatches = 5 # per side
    dim = (pad+swatchdim) * swatches
    img = PIL.Image.new("RGB", (dim, dim), "white")

    for i in range(min(swatches * swatches, len(all_out))):
        out = all_out[i]
        out = out.reshape(3, 64, 64)
        out = out.asnumpy() * 255
        out = np.uint8(out)
        out = np.moveaxis(out, 0, -1) # switch from channels_first to channels_last
        #print("check this: ")
        #print(out.shape)
        print(out.shape)
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
    while _classidx < 25:
        inp = gen_input_batch(nd.one_hot(nd.array([classes[x] % num_classes for x in range(_classidx,_classidx+train_generator.batch_size)]), num_classes) )
        _classidx+=train_generator.batch_size
        print(inp.shape)
        batch_out = gen(inp)
        #print(batch_out)
        print(batch_out.shape)
        for out in batch_out:
            all_out.append(out)
    render(all_out, filenum)

### some instrumentation
#render(train_generator.next()[0], -99)
sample(0)
train_generator.reset()
#exit()

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


###
### ugh
###

def get_lr(model):
    #return float(keras.backend.eval(model.optimizer.lr))
    return 1.0 # TODO

def set_lr(model, newlr):
    pass
    #model.optimizer.lr = keras.backend.variable(newlr)
    # TODO
    
###
### T R A I N
###

loss = mx.gluon.loss.HingeLoss()

batches = math.floor(train_generator.n / train_generator.batch_size)

disc_start_lr = get_lr(discrim)
disc_real_lr = get_lr(discrim)
disc_fake_lr = get_lr(discrim)

#adver_start_lr = get_lr(adver)
#adver_lr = get_lr(adver)

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
        y = real_y = nd.one_hot(nd.array(y), num_classes)
        #set_lr(discrim, disc_real_lr)
        
        with autograd.record():
            # real data loss
            real_output = discrim(nd.array(x))
            errD_real = loss(real_output, real_y)

            #disc_real_lr = abs(min(1.0, d_loss[0])) * disc_start_lr
        
            # get fake data
            #half_y = real_y[math.floor(len(real_y)/2):]            
            x_gen_input = gen_input_batch(real_y) # real classes with appended random noise inputs
            fake_x = gen(x_gen_input)
            y = nd.one_hot(nd.array([num_classes-1 for dummy in x]), num_classes) * 0.95
            #set_lr(discrim, disc_fake_lr)
            fake_output = discrim(fake_x.detach()) # why detach()?
            errD_fake = loss(fake_output, y)
            errD = (errD_real + errD_fake) * 0.5
            #print("loss REAL: {}, FAKE: {}".format(errD_real.mean(), errD_fake.mean()))
            errD.backward()

            #x = gen_input_batch() 
        
        trainerD.step(train_generator.batch_size)
        
        #y = np.array([inp[:num_classes] for inp in x])
        #set_lr(adver, adver_lr)
        with autograd.record():
            output = discrim(fake_x)
            errG = loss(output, real_y)
            #print("loss ADVR: {}".format(errG))
            x_gen_input = gen_input_batch(real_y) # same classes, different noise
            fake_x = gen(x_gen_input)
            #a_loss = adver.train_on_batch(x_gen_input, real_y)
            output = discrim(fake_x)
            errG2 = loss(output, real_y)
            #print("loss ADV2: {}".format(errG2))
            errG = (errG + errG2) * 0.5
            errG.backward()
            #print("loss ADVR: mean {}".format(errG.mean()))
            #print("ADVR: a_loss %f, %s %f @ %d/%d\n" % (a_loss[0], adver.metrics_names[1], a_loss[1], batch_num, batches))
            #adver_lr = abs(min(2.0, a_loss[0])) * adver_start_lr
        
        trainerG.step(train_generator.batch_size)
        
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
    
