#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <cmath>

#include "FixedPoint.h"
#include "Utilities.h"

//#define LUT // Enable for testing lut2

template<unsigned int IN, unsigned int OUT, typename T, typename U>
class DenseLayer {
private:
    U kernel[IN][OUT] __attribute__((aligned(64)));

public:
    explicit DenseLayer(const std::string &kernel_filename) {
        load_binary(kernel_filename, (U *) this->kernel, IN * OUT);

#ifdef LUT
        auto *integer_pointer = (int8_t *) kernel;

        for (unsigned row = 0; row < IN; row++) {
            for (unsigned column = 0; column < OUT; column++) {
                if (kernel[row][column] < Q4s3(-0.25f)) {
                    kernel[row][column] = Q4s3(-0.25f);
                }

                if (kernel[row][column] > Q4s3(0.125f)) {
                    kernel[row][column] = Q4s3(0.125f);
                }

                if (kernel[row][column] == Q4s3(-0.25f)) {
                    integer_pointer[row * OUT + column] = -1;
                }
                else if (kernel[row][column] == Q4s3(-0.125f)) {
                    integer_pointer[row * OUT + column] = -2;
                }
                else if (kernel[row][column] == Q4s3(0.125f)) {
                    integer_pointer[row * OUT + column] = 2;
                }
                else if (kernel[row][column] == Q4s3(0.0f)) {
                    integer_pointer[row * OUT + column] = 8;
                }
            }
        }
#endif
    }

    template<typename F>
    inline
    void apply(const T *input, T *output, F activation) {

#ifdef LUT
        for (unsigned i = 0; i < IN; i++) {
            for (unsigned j = 0; j < OUT; j++) {

                const auto kernelAbs = (uint8_t) abs(kernel[i][j].getRawData());
                const auto inputAbs = (uint8_t) abs(input[i].getRawData());

                auto result = (int8_t) ((inputAbs >> kernelAbs) + 1) >> 1;

                if(kernel[i][j] < U(0.0f))
                    result = -result;

                if(input[i] < T(0.0f))
                    result = -result;

                output[j] = output[j] + Q8s4((int8_t) result);
            }
        }

        for (unsigned i = 0; i < OUT; i++) {
            output[i] = activation(output[i]);
        }
#else
        for (unsigned i = 0; i < OUT; i++) {
            for (unsigned j = 0; j < IN; j++) {

                output[i] = output[i] + (input[j] * this->kernel[j][i]);
            }

            output[i] = activation(output[i]);
        }
#endif
    }
};

#endif //DENSELAYER_H
