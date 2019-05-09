import itertools

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
# noinspection PyPep8Naming
from keras import backend as tf

from references.polar.encoder import Encoder
from references.polar.sequence import Sequence

import simulation.utilities.channel as channel
import simulation.utilities.modulation as modulation
import simulation.utilities.demapper as demapper

# noinspection PyUnresolvedReferences
from models.layers import QuantizedDense


# noinspection PyPep8Naming
class NNDecoder:
    """Base class for NN decoders that supports real BPSK modulation.

     The model is based on the code and the paper: On Deep Learning-Based Channel Decoding, Tobias Gruber and
     Sebastian Cammerer and Jakob Hoydis and Stephan ten Brink: http://arxiv.org/abs/1701.07738
    """

    def __init__(self, N, K):
        """Initialize the decoder for the specific codeword and message size.

        The order of the decoder is used to calculate the codeword size and
        create the polar sequence.

        Args:
            N: Order of the decoder.
            K: Message length.

        Raises:
            ValueError: At least one argument is invalid.
        """
        if N > 1024 or N < 4:
            raise ValueError('Codeword size has to be in range [4, 1024]!')

        self._n = int(np.log2(N))
        self._N = N
        self._K = K

        if self._K > self._N:
            raise ValueError('Message size cannot exceed codeword size!')

        self.decoder = None

        self._unfrozen_positions = Sequence(self._N, self._K).unfrozen_positions

    def compose(self, layers, final_activation='sigmoid', constraint=None, use_bias=True):
        """Create the model of the NN according to the input design.

        Based on the input layer sizes the model of the decoder is composed so it can be trained
        in the subsequent step.

        Args:
            layers: array of integers where each element represents the size of one internal layer.
            final_activation: activation function in the last layer
            constraint: Constraints for kernel and weights
            use_bias: True if bias is used, false otherwise
        """
        self.decoder = Sequential()

        layer_type = Dense
        # layer_type = QuantizedDense

        decoder_layers = [layer_type(layers[0], activation='relu', kernel_constraint=constraint, bias_constraint=constraint,
                                use_bias=use_bias)]
        for size in layers[1:]:
            decoder_layers.append(layer_type(size, activation='relu', kernel_constraint=constraint,
                                        bias_constraint=constraint, use_bias=use_bias))
        decoder_layers.append(layer_type(self._N, activation=final_activation, kernel_constraint=constraint,
                                    bias_constraint=constraint, use_bias=use_bias))

        for layer in decoder_layers:
            self.decoder.add(layer)

        self.decoder.compile(optimizer='adam', loss='mse', metrics=[errors])

    def train(self, train_llrs=False):
        """Train the composed model of the decoder.

        Please note that the model has to be composed first.

        Args:
            train_llrs: If True input values will be converted to LLRs
        """
        nb_epoch = 2 ** 16
        batch_size = 256

        train_snr = 1
        train_snr_es = train_snr + 10 * np.log10(self._K / self._N)

        encoder = Encoder(self._N, self._K)

        messages = list(itertools.product([0, 1], repeat=self._K))

        encoded_messages = []
        inserted_messages = []
        for m in messages:
            encoded_messages.append(encoder.encode(list(m)))
            inserted_messages.append(encoder.bits_insertion(list(m)))

        demap = demapper.bpsk

        encoded_messages = np.array(encoded_messages)
        inserted_messages = np.array(inserted_messages)

        stddev = np.sqrt(1 / (2 * 10 ** (train_snr_es / 10)))

        for epoch in range(nb_epoch):

            modulated_message = modulation.bpsk(encoded_messages)
            noisy_message = np.real(channel.awgn(modulated_message, train_snr))
            if train_llrs:
                noisy_message = demap(noisy_message, stddev)

            self.decoder.fit(noisy_message, inserted_messages, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

    def decode(self, x, batch_size=256):
        """Perform decoding using given channel values.

        The function uses trained NN to decode the codeword.

        Args:
            x: Codeword to be decodes
            batch_size: Size of the input batch
        """
        decoded_codeword = np.round(self.decoder.predict(np.array([x]), batch_size)).astype(int)

        message = decoded_codeword[0][self._unfrozen_positions]

        return message


def errors(y_true, y_pred):
    """Metric used for the model in the loss function.

    Function calculates number of incorrectly decoded bits.
    """
    return tf.sum(tf.cast(tf.not_equal(y_true, tf.round(y_pred)), 'float32'))
