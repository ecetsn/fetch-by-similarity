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

// The print outs are given to analys the effect of noise on the marker and on the payload values

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
            double second_maxval = -1e20;
            int second_marker = -1;

            for (size_t ii = 0; ii < PAYLOAD_DIM; ii++) {
                double val = result_matrix[i + ii][j];
                if (val > maxval) {
                    second_maxval = maxval;
                    second_marker = marker;
                    maxval = val;
                    marker = static_cast<int>(ii);
                } else if (val > second_maxval) {
                    second_maxval = val;
                    second_marker = static_cast<int>(ii);
                }
            }

            // MAX_PAYLOAD_VAL = 256
            // Markers are 512
            if (maxval > MAX_PAYLOAD_VAL) {
                bool suspicious = (maxval < MAX_PAYLOAD_VAL * 1.4);

                double scale =
                    (MAX_PAYLOAD_VAL * 2 * PAYLOAD_PRECISION) /
                    result_matrix[i + marker][j];

                std::vector<int16_t> rec(PAYLOAD_DIM - 1);
                for (size_t k = 1; k < PAYLOAD_DIM; k++) {
                    auto idx = i + ((marker + k) % PAYLOAD_DIM);
                    rec[k - 1] =
                        static_cast<int16_t>(std::round(scale * result_matrix[idx][j]));
                }

                auto print_block = [&](std::string status) {
                    std::cerr << "BLOCK col = " << j
                              << " block = " << (i / PAYLOAD_DIM)
                              << " maxval = " << maxval
                              << " ratio = " << (maxval / 512.0)
                              << " status = " << status
                              << " (2nd max = " << second_maxval << " at index " << second_marker << ")"
                              << "\n";
                };

                if (suspicious) {
                    print_block("SUSPICIOUS");
                } else {
                    print_block("STRONG");
                }

                obtained_vals.push_back(rec);
            } else {
                 if (maxval > 10.0) { // arbitrary threshold to avoid printing every zero slot if needed
                     std::cerr << "BLOCK col = " << j
                               << " block = " << (i / PAYLOAD_DIM)
                               << " maxval = " << maxval
                               << " ratio = " << (maxval / 512.0)
                               << " status = SKIP"
                               << " (2nd max = " << second_maxval << " at index " << second_marker << ")"
                               << "\n";
                 }
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