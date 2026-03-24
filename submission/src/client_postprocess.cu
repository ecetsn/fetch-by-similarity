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

// -----------------------------------------------------------------------------
// Helper functions for post-processing
// -----------------------------------------------------------------------------

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

static std::vector<std::vector<int16_t>> decode_results(const std::vector<double>& slots, int n_cols) {
    auto result_matrix = to_matrix_form_local({slots}, n_cols);
    std::vector<std::vector<int16_t>> obtained_vals;
    for (int j = 0; j < n_cols; j++) {
        for (size_t i = 0; i < result_matrix.size(); i += PAYLOAD_DIM) {
            int marker = -1;
            double maxval = 0;
            for (size_t ii = 0; ii < PAYLOAD_DIM; ii++) {
                if (result_matrix[i + ii][j] > maxval) {
                    maxval = result_matrix[i + ii][j];
                    marker = ii;
                }
            }
            if (maxval > MAX_PAYLOAD_VAL) {
                double scale = (MAX_PAYLOAD_VAL * 2 * PAYLOAD_PRECISION) / result_matrix[i + marker][j];
                std::vector<int16_t> rec(PAYLOAD_DIM - 1);
                for (size_t k = 1; k < PAYLOAD_DIM; k++) {
                    auto idx = i + ((marker + k) % PAYLOAD_DIM);
                    rec[k - 1] = static_cast<int16_t>(std::round(scale * result_matrix[idx][j]));
                }
                obtained_vals.push_back(rec);
            }
        }
    }
    std::sort(obtained_vals.begin(), obtained_vals.end());
    return obtained_vals;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    int size_int = std::stoi(argv[1]);
    bool count_only = (argc > 2 && std::string(argv[2]) == "--count_only");
    InstanceParams prms(static_cast<InstanceSize>(size_int), count_only);
    setup_he_context(prms.getSize());
    auto all_slots = read2vecs<double>(prms.encdir() / "raw-result.bin", prms.getNSlots());

    if (count_only) {
        double total_count = 0;
        if (!all_slots.empty() && !all_slots[0].empty()) {
            total_count = all_slots[0][0];
        }
        long count = static_cast<long>(std::max(0.0, std::round(total_count)));
        write2disk<long>(prms.iodir() / "results.bin", {{count}});
    } else {
        std::vector<std::vector<int16_t>> all_res;
        for (const auto& slots : all_slots) {
            auto res = decode_results(slots, prms.getNCols());
            all_res.insert(all_res.end(), res.begin(), res.end());
        }
        std::sort(all_res.begin(), all_res.end());
        write2disk<int16_t>(prms.iodir() / "results.bin", all_res);
    }

    return 0;
}
