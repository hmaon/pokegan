from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.nn import sigmoid


# SWISH is f(x) = x * sigmoid(beta * x)
# see https://arxiv.org/abs/1710.05941

# Alternate form from wolframalpha is: x - x/(e^(0.5 b x) + 1)


class SWISH(Layer):
    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(SWISH, self).__init__(**kwargs)

    def build(self, input_shape):
        # trainable beta
        
        # shape=input_shape[1:] allocates a weight for each unit which is unhelpful and memory-hungry
        # it really needs shared_axes, like PReLU() so that's a TODO
        self.beta = self.add_weight(name='beta', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(SWISH, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x * K.sigmoid(x * self.beta)

    def compute_output_shape(self, input_shape):
        return input_shape