# noinspection PyPep8Naming
import keras.backend as K
from keras.layers.core import Dense

import tensorflow as tf


class QuantizedDense(Dense):
    """Quantized version of the dense layer used for the quantization aware training.

    The layer uses the fake quantization to quantize values with 4 bits and clips
    the values in range [-0.25, 0.125] so LUT mapping can be used.
    """

    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

    def call(self, inputs):
        fake_quantize = tf.quantization.fake_quant_with_min_max_vars

        quantized_kernel = fake_quantize(self.kernel, -1, 0.9921875, 8)  # int8 Q8.7
        # quantized_kernel = fake_quantize(self.kernel, -1, 0.875, 4)  # int4/lut2 Q4.3
        # quantized_kernel = tf.clip_by_value(quantized_kernel, -0.25, 0.125)  # Only for lut2

        quantized_inputs = fake_quantize(inputs, -8, 7.9375, 8)  # Q8.4

        output = K.dot(quantized_inputs, quantized_kernel)
        output = fake_quantize(output, -8, 7.9375, 8)  # Q8.4

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
            output = fake_quantize(output, -8, 7.9375, 8)  # Q8.4
        if self.activation is not None:
            output = self.activation(output)
            output = fake_quantize(output, -8, 7.9375, 8)  # Q8.4
        return output
