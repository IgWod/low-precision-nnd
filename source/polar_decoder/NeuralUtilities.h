#ifndef NEURALUTILITIES_H
#define NEURALUTILITIES_H

#include <cstdint>

template <typename T>
T relu(const T input) {
    if (input > T(0.0f)) {
        return input;
    } else {
        return T(0.0f);
    }
}

template<typename T>
T hard_sigmoid(const T input) {
    if (input < T(-2.5f)) {
        return T(0.0f);
    } else if (input > T(2.5f)) {
        return T(1.0f);
    } else {
        return T(0.5f) + (input * T(0.2f));
    }
}

template<typename T>
int8_t make_decision(const T input) {
    if (input > T(0.5f)) {
        return 1;
    } else {
        return 0;
    }
}

#endif // NEURALUTILITIES_H
