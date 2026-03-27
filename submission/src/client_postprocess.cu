// client_postprocess.cu - Client post-processing of encrypted results (HEonGPU)
//============================================================================
// Copyright (c) 2025, Amazon Web Services
// All rights reserved.
//
// This software is licensed under the terms of the Apache License v2.
// See the file LICENSE.md for details.
//============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "params.cuh"
#include "utils.cuh"

using namespace heongpu;

// Values are comming as [512, p1 / precision, p2 / precision, ...]

static std::vector<std::vector<double>> to_matrix_form_local(const std::vector<std::vector<double>>& slots, size_t n_cols) {
    if (slots.empty() || slots[0].empty()) return {};
    int n_rows_per_vector = static_cast<int>(slots[0].size() / n_cols);
    std::vector<std::vector<double>> matrix(slots.size() * n_rows_per_vector, std::vector<double>(n_cols));
    for (size_t i = 0; i < matrix.size(); i++) {
        size_t slots_i = i % slots.size();
        size_t slots_j = n_cols * (i / slots.size());
        for (size_t j = 0; j < n_cols; j++, slots_j++) {
            matrix[i][j] = slots[slots_i][slots_j];
        }
    }
    return matrix;
}

static std::vector<std::vector<int16_t>>
decode_results(const std::vector<double>& slots, int n_cols) {
    auto result_matrix = to_matrix_form_local({slots}, n_cols);
    std::vector<std::vector<int16_t>> obtained_vals;

    for (int j = 0; j < n_cols; j++) {
        for (size_t i = 0; i < result_matrix.size(); i += PAYLOAD_DIM) {
            int marker = -1;
            double maxval = -1e20;

            for (size_t ii = 0; ii < PAYLOAD_DIM; ii++) {
                double val = result_matrix[i + ii][j];
                if (val > maxval) {
                    maxval = val;
                    marker = static_cast<int>(ii);
                }
            }
            // MAX_PAYLOAD_VAL = 256
            // Markers wihtout scale (so maxval can be) : MAX_PAYLOAD_VAL * 2
            if (maxval > MAX_PAYLOAD_VAL) {
                // Expected marker value (reference style)
                double expected_marker = MAX_PAYLOAD_VAL * 2;

                // Reject weak / suspicious marker
                if (maxval < MAX_PAYLOAD_VAL * 1.4) {
                    std::cerr << "BLOCK col = " << j
                            << " block = " << (i / PAYLOAD_DIM)
                            << ", maxval = " << maxval
                            << ", ratio = " << (maxval / expected_marker) << "\n";
                    std::stringstream ss;
                    for (size_t k = 0; k < PAYLOAD_DIM; k++) {
                        auto x = result_matrix[i + k][j];
                        ss << x << ' ';
                    }
                    throw(std::runtime_error(
                        "Marker not found in payload: [" + ss.str() + "]"));
                }
#ifdef DEBUG
                std::cout << "BLOCK col = " << j
                        << " block = " << (i / PAYLOAD_DIM)
                        << ", maxval = " << maxval
                        << ", ratio = " << (maxval / expected_marker)
                        << "\n";
#endif
                // Scaling (M / M')
                double scale = (MAX_PAYLOAD_VAL * 2 * PAYLOAD_PRECISION)  
                                / (result_matrix[i + marker][j]);

                std::vector<int16_t> rec(PAYLOAD_DIM - 1);
                for (size_t k = 1; k < PAYLOAD_DIM; k++) {
                    auto idx = i + ((marker + k) % PAYLOAD_DIM);
                    rec[k - 1] =
                        static_cast<int16_t>(std::round(scale * result_matrix[idx][j]));
                }

                obtained_vals.push_back(rec);
            }
        }
    }

    std::sort(obtained_vals.begin(), obtained_vals.end());
    return obtained_vals;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    bool count_only = (argc > 2 && std::string(argv[2]) == "--count_only");

    InstanceParams prms(size, count_only);
    setup_he_context(prms.getSize());

    auto vs = read2vecs<double>(prms.encdir() / "raw-result.bin", prms.getNSlots());
    assert(vs.size() == 1);
    auto slots = vs[0];

    if (count_only) {
        long count = std::round(slots[0]);
        write2disk<long>(prms.iodir() / "results.bin", {{count}});
    } else {
        auto res = decode_results(slots, prms.getNCols());
        write2disk<int16_t>(prms.iodir() / "results.bin", res);
    }

    return 0;
}