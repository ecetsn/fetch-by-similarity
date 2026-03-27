// client_decrypt_decode.cu - Client decrypt and decode results (HEonGPU)
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
        load_from_file_raw<HEContextImpl<Scheme::CKKS>>((prms.keydir() / "cc.bin").string()));
    auto sk = load_from_file_raw<Secretkey<Scheme::CKKS>>((prms.keydir() / "sk.bin").string());

    HEDecryptor<Scheme::CKKS> decryptor(context, sk);
    HEEncoder<Scheme::CKKS> encoder(context);

    std::string input_file = (prms.encdir() / "results.bin").string();
    
    std::vector<Ciphertext<Scheme::CKKS>> result_cts;
    
    result_cts = load_batch<Scheme::CKKS>(input_file, context);

    std::vector<std::vector<double>> all_slots;
    for (size_t i = 0; i < result_cts.size(); ++i) {
        auto& ct = result_cts[i];
 #ifdef DEBUG 
        std::cout << "[debug] Decrypting CT " << i << " depth=" << ct.depth() 
                  << ", level=" << ct.level() << std::endl;
 #endif
        Plaintext<Scheme::CKKS> pt(context);
        decryptor.decrypt(pt, ct);
        std::vector<double> slots;
        encoder.decode(slots, pt);
        all_slots.push_back(slots);
    }
    
    write2disk<double>(prms.encdir() / "raw-result.bin", all_slots);

 #ifdef DEBUG   // Print first few values
    std::cout << "[debug] First 10 values from raw-result.bin: " << std::endl;
    for (int i = 0; i < std::min((size_t)10, all_slots[0].size()); ++i) {
        std::cout << "[debug] all_slots[0][" << i << "] = " << all_slots[0][i] << std::endl;
    }
#endif
    return 0;
}
