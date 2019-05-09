#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <papi.h>

#include "FixedPoint.h"
#include "NeuralDecoder.h"

#include "DecoderMain.h"

//#define ENABLE_PROFILING // Enable for PAPI counters

using namespace std::chrono;

using DecoderType = Q8s4;
using WeightsType = Q8s7; // For int8
//using WeightsType = Q4s3; // For int4 and lut2

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cout << "Not enough arguments passed!" << std::endl;
        std::cout << "Please specify test directory!" << std::endl;
        return -1;
    }

    auto decoder = NeuralDecoder<DecoderType, WeightsType>("data/weights/" + std::string(argv[1]) + "/");

    DecoderType input[16] __attribute__((aligned(64))) = {};
    DecoderType output[16] __attribute__((aligned(64))) = {};

    float golden[16] = {};

    unsigned msg_pos[8] = {6, 10, 12, 7, 11, 13, 14, 15};

    unsigned long total_duration = 0;

    std::string files_snr[] = {"1.000000", "1.795645", "2.671589", "3.645881", "4.743429", "6.000000"};

    constexpr unsigned nb_papi_events = 1;
    long long papi_counters[nb_papi_events];
    int papi_events[nb_papi_events] = {PAPI_L1_TCM};

    constexpr unsigned k_max_file_index = 6;
    constexpr unsigned k_number_iter = 10000; // 100K max

    std::vector<long long> counter_results(k_number_iter);

    for (const auto &file_snr : files_snr) {

        unsigned errors = 0;

        std::string path = "data/tests/float32/";
        std::string file_ending = file_snr + "-dB.bin";

        std::string llrs_filename = path + "in-16-8-";
        llrs_filename += file_ending;
        std::string msg_filename = path + "msg-16-8-";
        msg_filename += file_ending;

        std::ifstream llrs(llrs_filename, std::ios::in | std::ios::binary);
        std::ifstream messages(msg_filename, std::ios::in | std::ios::binary);

        for (unsigned i = 0; i < k_number_iter; i++) {

            for (auto &in : input) {
                float read_value;
                llrs.read((char *) &read_value, sizeof(float));
                in = DecoderType(read_value);
            }

            messages.read((char *) golden, 4 * 8);
            std::fill_n(output, 16, DecoderType(0.0f));

#ifdef ENABLE_PROFILING
            // Warm-up cache
            decoder.decode(input, output);
#endif

            high_resolution_clock::time_point t1 = high_resolution_clock::now();

#ifdef ENABLE_PROFILING
            if (PAPI_start_counters(papi_events, nb_papi_events) != PAPI_OK)
                std::cout << "PAPI has not started!" << std::endl;
            for(unsigned internal = 0; internal < 100; internal++) {
#endif

            decoder.decode(input, output);

#ifdef ENABLE_PROFILING
            }
            if ( PAPI_stop_counters(papi_counters, nb_papi_events) != PAPI_OK)
                std::cout << "PAPI has not stopped!" << std::endl;
#endif

            high_resolution_clock::time_point t2 = high_resolution_clock::now();

            total_duration += duration_cast<microseconds>(t2 - t1).count();

            counter_results.at(i) = papi_counters[0];

            for (unsigned idx = 0; idx < 8; idx++) {
                if (make_decision(output[msg_pos[idx]]) != (int8_t) golden[idx]) {
                    errors++;
                    break;
                }
            }
        }

        llrs.close();
        messages.close();

        std::cout << "Final BLER for file " << llrs_filename << " is " << errors / (double) k_number_iter << std::endl;
    }

    const auto sum = std::accumulate(counter_results.begin(), counter_results.end(), 0L);

    const auto mean = sum / (double) k_number_iter;

    auto stddev_accumulator = 0.0;
    for (auto r : counter_results)
        stddev_accumulator += pow(r - mean, 2);

    const auto stddev = sqrt((stddev_accumulator) / (k_number_iter - 1));

#ifdef ENABLE_PROFILING
    std::cout << "PAPI Counter mean value (per 100 iterations): " << mean << std::endl;
    std::cout << "PAPI Counter stddev value (per 100 iterations): " << stddev << std::endl;
#endif

    std::cout << "Average execution time: " << total_duration / (k_max_file_index * k_number_iter * 100) << "us" << std::endl;

    return 0;
}
