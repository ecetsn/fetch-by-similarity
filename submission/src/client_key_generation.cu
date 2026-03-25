#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <heongpu/heongpu.hpp>

#include "params.cuh"
#include "running_sums.cuh"
#include "slot_replication.cuh"
#include "utils.cuh"

using namespace heongpu;

namespace {

constexpr int CKKS_SCALING_MOD_BITS = 42;
constexpr int CKKS_FIRST_MOD_BITS   = 57;
constexpr int MULT_DEPTH = 26;

std::vector<int> special_primes_bits() {
    return {60, 60, 60};
}

std::vector<int> build_q_bits(int mult_depth) {
    std::vector<int> bits;
    bits.reserve(mult_depth + 1);
    bits.push_back(CKKS_FIRST_MOD_BITS);
    for (int i = 0; i < mult_depth; ++i) {
        bits.push_back(CKKS_SCALING_MOD_BITS);
    }
    return bits;
}

int sum_bits(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 0);
}
}

/*
Recommended Modulus Sizes for 128-bit Security

 Polynomial Degree (N)      Total Modulus Bit-Length (log2 Q)
 -----------------------------------------------------------
 2^12  (4096)               ~109 bits
 2^13  (8192)               ~218 bits
 2^14  (16384)              ~438 bits
 2^15  (32768)              ~881 bits
 2^16  (65536)              ~1750 bits
*/

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

    const heongpu::sec_level_type sec =
        (prms.getSize() == InstanceSize::TOY) ? heongpu::sec_level_type::none
                                              : heongpu::sec_level_type::sec128;

    const int mult_depth = MULT_DEPTH;

    // maybe also different depth for different count_only?

    auto q_bits = build_q_bits(mult_depth);
    auto sp_bits = special_primes_bits();

    auto context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(sec);
    context->set_poly_modulus_degree(prms.getRingDim());
    context->set_coeff_modulus_bit_sizes(q_bits, sp_bits);
    context->generate();

    context->print_parameters();
    std::cout << "Total modulus bit-length: " << sum_bits(q_bits) << "\n";

    HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);

    Secretkey<heongpu::Scheme::CKKS> secret_key(context);
    keygen.generate_secret_key(secret_key);

    Publickey<heongpu::Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    Relinkey<heongpu::Scheme::CKKS> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    // Identify all required rotation amounts for Galois Keys
    std::vector<int> all_rots;

    // 1. Rotations for slot replication (DFS tree)
    auto rots4reps = DFSSlotReplicator::get_rotation_amounts(prms.getDegrees());
    all_rots.insert(all_rots.end(), rots4reps.begin(), rots4reps.end());

    if (!count_only) {
        // 2. Rotations for payload extraction/shifts
        for (int i = 1; i < PAYLOAD_DIM; i++) {
            all_rots.push_back(-i * prms.getNCols());
        }
        
        // 3. Rotations for running sums (match indexing)
        auto shifts2 = RunningSums::get_shift_amounts(prms.getNSlots(), prms.getNCols(), RUNNING_SUM_LEVELS);
        all_rots.insert(all_rots.end(), shifts2.begin(), shifts2.end());

        // 4. Rotations for total sums (replication of payloads across each column).
        // total_sums uses POSITIVE rotation amounts (left cyclic shift), matching
        // HEonGPU convention confirmed from examples.
        auto sum_rows_stride = prms.getNCols() * PAYLOAD_DIM;
        int log_stride = static_cast<int>(std::log2(prms.getNSlots() / sum_rows_stride));
        for (int i = 0; i < log_stride; ++i) {
            all_rots.push_back((sum_rows_stride) << i); 
        }
    } else {
       // Rotations for global count accumulation
       int log_slots = static_cast<int>(std::log2(prms.getNSlots()));
       for (int i = 0; i < log_slots; i++) {
           all_rots.push_back(-(1 << i));
       }
    }

    // Deduplicate rotation amounts
    std::sort(all_rots.begin(), all_rots.end());
    all_rots.erase(std::unique(all_rots.begin(), all_rots.end()), all_rots.end());

    Galoiskey<heongpu::Scheme::CKKS> galois_key(context, all_rots);
    keygen.generate_galois_key(galois_key, secret_key);

    std::filesystem::create_directories(prms.keydir());
    save_to_file_raw(*context, (prms.keydir() / "cc.bin").string());
    save_to_file_raw(public_key, (prms.keydir() / "pk.bin").string());
    save_to_file_raw(secret_key, (prms.keydir() / "sk.bin").string());
    save_to_file_raw(relin_key,  (prms.keydir() / "mk.bin").string());
    save_to_file_raw(galois_key, (prms.keydir() / "rk.bin").string());

    std::cout << "Key generation completed. " << all_rots.size()
              << " rotation keys generated.\n";

    return 0;
}