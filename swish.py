from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.nn import sigmoid
from tensorflow.python.keras.engine.base_layer import InputSpec


# SWISH is f(x) = x * sigmoid(beta * x)
# see https://arxiv.org/abs/1710.05941

# Alternate form from wolframalpha is: x - x/(e^(0.5 b x) + 1)


class SWISH(Layer):
    def __init__(self, shared_axes=None, trainable_beta=True, beta_initializer="ones", **kwargs):
        # self.output_dim = output_dim
        super(SWISH, self).__init__(**kwargs)
        self.trainable_beta = True
        self.beta_initializer = beta_initializer
        # from PReLU:
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        # from PReLU:
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1

        # trainable beta
        self.beta = self.add_weight(name="beta", shape=param_shape, initializer=self.beta_initializer, trainable=self.trainable_beta)
        # also from PReLU:
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, x):
        return x * K.sigmoid(x * self.beta)

    def compute_output_shape(self, input_shape):
        return input_shape
