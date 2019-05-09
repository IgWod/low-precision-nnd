#ifndef UTILITIES_H
#define UTILITIES_H

#include <fstream>

template <typename T>
void load_binary(const std::string &filename, T *buffer, const unsigned size) {
    std::fstream stream(filename, std::ios::in | std::ios::binary);

    for (int byte_nb = 0; byte_nb < size; byte_nb++) {
        float read_value;
        stream.read((char *) &read_value, sizeof(float));
        buffer[byte_nb] = T(read_value);
    }

    stream.close();
}

#endif //UTILITIES_H
