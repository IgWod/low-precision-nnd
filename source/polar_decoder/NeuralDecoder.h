#ifndef NEURAL_DECODER_H
#define NEURAL_DECODER_H

#include "NeuralUtilities.h"

#include "DenseLayer.h"

constexpr unsigned k_first_layer_size = 512;
constexpr unsigned k_second_layer_size = 256;
constexpr unsigned k_third_layer_size = 128;

template<typename T, typename U>
class NeuralDecoder {
private:
    DenseLayer<16, k_first_layer_size, T, U> input_layer;
    DenseLayer<k_first_layer_size, k_second_layer_size, T, U> first_hidden_layer;
    DenseLayer<k_second_layer_size, k_third_layer_size, T, U> second_hidden_layer;
    DenseLayer<k_third_layer_size, 16, T, U> output_layer;
public:
    explicit NeuralDecoder(const std::string &filename) : input_layer(DenseLayer<16, k_first_layer_size, T, U>(filename + "input")),
                                                          first_hidden_layer(
                                                              DenseLayer<k_first_layer_size, k_second_layer_size, T, U>(filename + "first_hidden")),
                                                          second_hidden_layer(
                                                              DenseLayer<k_second_layer_size, k_third_layer_size, T, U>(filename + "second_hidden")),
                                                          output_layer(
                                                              DenseLayer<k_third_layer_size, 16, T, U>(filename + "output")) {}

    void decode(const T *input, T *output) {
        T temp_buffer_1[k_first_layer_size] __attribute__((aligned(64)));
        T temp_buffer_2[k_second_layer_size] __attribute__((aligned(64)));

        std::fill_n(temp_buffer_1, k_first_layer_size, T(0.0f));
        this->input_layer.apply(input, temp_buffer_1, relu<T>);

        std::fill_n(temp_buffer_2, k_second_layer_size, T(0.0f));
        this->first_hidden_layer.apply(temp_buffer_1, temp_buffer_2, relu<T>);

        std::fill_n(temp_buffer_1, k_third_layer_size, T(0.0f));
        this->second_hidden_layer.apply(temp_buffer_2, temp_buffer_1, relu<T>);

        std::fill_n(output, 16, T(0.0f));
        this->output_layer.apply(temp_buffer_1, output, hard_sigmoid<T>);
    }
};

#endif //NEURAL_DECODER_H
