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
#include "utils.cuh"

using namespace heongpu;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " instance-size\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    int size_int = std::stoi(argv[1]);
    bool count_only = (argc > 2 && std::string(argv[2]) == "--count_only");
    InstanceParams prms(static_cast<InstanceSize>(size_int), count_only);
    setup_he_context(prms.getSize());

    cudaSetDevice(0);

    // Load context and secret key
    auto context = std::make_shared<HEContextImpl<Scheme::CKKS>>(
        heongpu::serializer::load_from_file<HEContextImpl<Scheme::CKKS>>((prms.keydir() / "cc.bin").string()));
    auto sk = heongpu::serializer::load_from_file<Secretkey<Scheme::CKKS>>((prms.keydir() / "sk.bin").string());

    HEDecryptor<Scheme::CKKS> decryptor(context, sk);
    HEEncoder<Scheme::CKKS> encoder(context);

    std::string input_file = (prms.encdir() / "results.bin").string();
    std::cout << "[debug] loading from " << input_file << std::endl;
    
    std::vector<Ciphertext<Scheme::CKKS>> result_cts;
    try {
        result_cts = load_batch<Scheme::CKKS>(input_file, context);
    } catch (...) {
        // Fallback for single ciphertext (e.g. query.bin or old results.bin)
        Ciphertext<Scheme::CKKS> ct(context);
        load_ciphertext(ct, input_file);
        result_cts.push_back(std::move(ct));
    }
    
    std::vector<std::vector<double>> all_slots;
    for (auto& ct : result_cts) {
        Plaintext<Scheme::CKKS> pt(context);
        decryptor.decrypt(pt, ct);
        std::vector<double> slots;
        encoder.decode(slots, pt);
        all_slots.push_back(slots);
    }
    
    write2disk<double>(prms.encdir() / "raw-result.bin", all_slots);
    
    if (!all_slots.empty()) {
        std::cout << "[debug] Decrypted first 10 similarity values of first batch:" << std::endl;
        for (int i = 0; i < 10 && i < all_slots[0].size(); i++) {
            std::cout << "  slot[" << i << "]: " << all_slots[0][i] << std::endl;
        }
    }
    
    if (count_only) {
        // Assuming count_only implies a single ciphertext result, or that the sum is in the first slot of the first batch
        if (!all_slots.empty() && !all_slots[0].empty()) {
            std::cout << "[debug] sum of all slots: " << std::round(all_slots[0][0]) << std::endl;
        } else {
            std::cout << "[debug] No slots to sum for count_only mode." << std::endl;
        }
    }

    return 0;
}
