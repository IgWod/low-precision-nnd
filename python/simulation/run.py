import numpy as np
import os
import matplotlib.pyplot as plt

import simulation.utilities.modulation
import simulation.utilities.channel
import simulation.utilities.demapper

from references.polar.encoder import Encoder
from references.polar.scdecoder import SCDecoder

from models.nn_decoder import NNDecoder, errors

from keras.models import load_model
from keras.constraints import min_max_norm


def get_modulation(modulation):
    if modulation == 'bpsk':
        return simulation.utilities.modulation.bpsk
    else:
        raise ValueError('Unsupported modulation scheme! Supported modulations: BPSK!')


def get_demapper(modulation):
    if modulation == 'bpsk':
        return simulation.utilities.demapper.bpsk
    else:
        raise ValueError('Unsupported demapper scheme! Supported demappers: BPSK!')


def get_channel(channel):
    if channel == 'awgn':
        return simulation.utilities.channel.awgn
    else:
        raise ValueError('Unsupported channel type! Supported channels: AWGN!')


# noinspection PyPep8Naming
def get_decoder(decoder, N, K):
    if decoder == 'SC':
        return SCDecoder(N, K), True
    else:
        try:
            data_path = os.environ['DEEPRAN_DATA_PATH']
        except KeyError:
            data_path = '/home/iwodiany/Projects/DeepRAN/data/'

        model_name = '%s-%s-%s.h5' % (decoder, N, K)
        model_path = data_path + 'models/' + model_name

        nn_decoder = NNDecoder(N, K)

        if os.path.isfile(model_path):
            nn_decoder.decoder = load_model(model_path, custom_objects={'errors': errors})

        if decoder == 'NN':
            if nn_decoder.decoder is None:
                nn_decoder.compose([512, 256, 128])
                nn_decoder.train(train_llrs=True)
                nn_decoder.decoder.save(model_path)
            return nn_decoder, True
        elif decoder == 'NN_LLR':
            if nn_decoder.decoder is None:
                nn_decoder.compose([512, 256, 128], final_activation='hard_sigmoid', constraint=min_max_norm(-1.0, 1.0),
                                   use_bias=False)
                nn_decoder.train(train_llrs=True)
                nn_decoder.decoder.save(model_path)
            return nn_decoder, True
        else:
            raise ValueError('Unsupported decoder type! Supported decoders: SC, NN, NN_LLR!')


# noinspection PyPep8Naming
def run_simulation(N, K, decoders, iters_per_snr=1000, snr_range=None, modulation='bpsk', channel='awgn',
                   save_data=False, load_path=None):
    """Run simulation for given decoders in the specific SNR range.

    The function takes list of decoders and codeword parameters as well as SNR range and channel parameters and runs
    the simulation in each SNR requested number of times and plots the result on the graph. The function is used
    to evaluate the wireless performance of the decoders.

    Args:
        N - codeword length.
        K - message length.
        decoders - list of decoders to run
        iters_per_snr - Number of trials per SNR.
        snr_range - An array in the given format [start SNR, end SNR, number of steps].
        modulation - Modulation scheme to be used for testing.
        channel - Channel to be used for testing
        save_data - Save test data to the file
        load_path - Path for loading test data
    """

    if snr_range is None:
        snr_start = 1
        snr_end = 6
        snr_points = 6
    else:
        snr_start = snr_range[0]
        snr_end = snr_range[1]
        snr_points = snr_range[2]

    modulate = get_modulation(modulation)
    add_noise = get_channel(channel)
    demap = get_demapper(modulation)
    encoder = Encoder(N, K)

    snr_start_es = snr_start + 10 * np.log10(K / N)
    snr_stop_es = snr_end + 10 * np.log10(K / N)

    stddev_start = np.sqrt(1 / (2 * 10 ** (snr_start_es / 10)))
    stddev_stop = np.sqrt(1 / (2 * 10 ** (snr_stop_es / 10)))

    stddevs = np.linspace(stddev_start, stddev_stop, snr_points)

    Eb_N0 = 10 * np.log10(1 / (2 * stddevs ** 2)) - 10 * np.log10(K / N)

    print("Starting simulation for parameters N=%d, K=%d in SNR range [%.2f, %.2f] in %s channel with %s modulation"
          % (N, K, snr_start, snr_end, channel, modulation))

    for decoder_name in decoders:

        (decoder, llrs_enabled) = get_decoder(decoder_name, N, K)

        nb_errors = np.zeros(len(stddevs), dtype=int)

        for iter_snr in range(len(stddevs)):

            if load_path is None:
                messages = []
                codewords = []
                for _ in range(iters_per_snr):
                    message = np.random.randint(0, 2, size=K)
                    codeword = encoder.encode(message)

                    messages.append(message)
                    codewords.append(codeword)

                messages = np.array(messages)
                codewords = np.array(codewords)

                modulated_symbols = modulate(codewords)
                noisy_data = add_noise(modulated_symbols, stddevs[iter_snr])

                if llrs_enabled:
                    noisy_data = demap(noisy_data, stddevs[iter_snr])

                if save_data:
                    messages.astype(np.float32).tofile('msg-%d-%d-%f-dB.bin' % (N, K, Eb_N0[iter_snr]))
                    noisy_data.astype(np.float32).tofile('in-%d-%d-%f-dB.bin' % (N, K, Eb_N0[iter_snr]))
            else:
                messages = np.fromfile(load_path + 'msg-%d-%d-%f-dB.bin' % (N, K, Eb_N0[iter_snr]), dtype=np.float32) \
                    .reshape((iters_per_snr, 8))
                noisy_data = np.fromfile(load_path + 'in-%d-%d-%f-dB.bin' % (N, K, Eb_N0[iter_snr]), dtype=np.float32) \
                    .reshape((iters_per_snr, 16))

            print("Running %s decoder in %.2fdB Eb/N0" % (decoder_name, Eb_N0[iter_snr]))

            for iter_number in range(iters_per_snr):

                decoded_message = decoder.decode(noisy_data[iter_number])

                # noinspection PyUnresolvedReferences
                if (messages[iter_number] != decoded_message).any():
                    nb_errors[iter_snr] += 1
                    print("In iteration %d decoding failed! Current BLER is %.10f"
                          % (iter_number, nb_errors[iter_snr] / (iter_number + 1)))

            print("Final BLER for %.2fdB Eb/N0 is %.10f" % (Eb_N0[iter_snr], nb_errors[iter_snr] / iters_per_snr))

        plt.plot(Eb_N0, nb_errors / iters_per_snr)

    plot_title = 'BLER in %s channel with %s modulation' % (channel.upper(), modulation.upper())

    plt.legend(decoders, loc=3)
    plt.title(plot_title)
    plt.yscale('log')
    plt.yticks([1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
    plt.ylabel('BLER')
    plt.xlabel('$E_b/N_0$')
    plt.grid(True)
    plt.show()


def save_kernels(model, filenames=None):
    """Helper function used to extract weights to C++.

    """
    if filenames is None:
        filenames = ['input', 'first_hidden', 'second_hidden', 'output']

    if len(model.layers) != len(filenames):
        raise ValueError('Not enough destinations!')

    for (layer, filename) in zip(model.layers, filenames):
        layer.get_weights()[0].astype(np.float32).tofile(filename)
