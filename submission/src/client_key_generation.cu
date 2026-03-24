#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <heongpu/heongpu.hpp>

#include "params.cuh"
#include "running_sums.cuh"
#include "slot_replication.cuh"
#include "utils.cuh"

using namespace heongpu;

/**
 * client_key_generation:
 * This executable generates the FHE keys (Secret, Public, Relinearization, and Galois)
 * required for the fetch-by-similarity workload.
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

    // Security level: none for TOY (testing), sec128 for others
    heongpu::sec_level_type sec = (prms.getSize() == InstanceSize::TOY)
        ? heongpu::sec_level_type::none
        : heongpu::sec_level_type::sec128;

    // Build modulus chain bit sizes
    auto q_bits = build_q_bits(prms.getMultDepth());
    auto context = heongpu::GenHEContext<Scheme::CKKS>(sec);
    context->set_poly_modulus_degree(prms.getRingDim());
    context->set_coeff_modulus_bit_sizes(q_bits, get_special_primes_bits());
    context->generate();

    HEKeyGenerator<Scheme::CKKS> keygen(context);
    
    // Generate Secret Key
    Secretkey<Scheme::CKKS> secret_key(context);
    keygen.generate_secret_key(secret_key);
    
    // Generate Public Key
    Publickey<Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);
    
    // Generate Relinearization Key (used for multiplications)
    Relinkey<Scheme::CKKS> relin_key(context);
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
            all_rots.push_back((sum_rows_stride) << i);  // positive = left shift
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

    // Generate Galois Keys
    Galoiskey<Scheme::CKKS> galois_key(context, all_rots);
    keygen.generate_galois_key(galois_key, secret_key);

    // Serialize keys to disk
    std::filesystem::create_directories(prms.keydir());
    heongpu::serializer::save_to_file(*context, (prms.keydir() / "cc.bin").string());
    heongpu::serializer::save_to_file(public_key, (prms.keydir() / "pk.bin").string());
    heongpu::serializer::save_to_file(secret_key, (prms.keydir() / "sk.bin").string());
    heongpu::serializer::save_to_file(relin_key, (prms.keydir() / "mk.bin").string()); // "mk" in benchmark terminology
    heongpu::serializer::save_to_file(galois_key, (prms.keydir() / "rk.bin").string()); // "rk" in benchmark terminology

    std::cout << "Key generation completed. " << all_rots.size() << " rotation keys generated." << std::endl;

    return 0;
}
