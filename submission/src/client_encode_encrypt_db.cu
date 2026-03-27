// client_encode_encrypt_db.cu - Encrypting the dataset (HEonGPU)
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
#include <iomanip>
#include <sstream>

#include <cuda_runtime.h>
#include <heongpu/heongpu.hpp>

#include "params.cuh"
#include "utils.cuh"

using namespace heongpu;
namespace fs = std::filesystem;

void add_markers(std::vector<std::vector<int16_t>>& payloads) {
    for (auto& p : payloads) {
        p.insert(p.begin(), 2 * MAX_PAYLOAD_VAL * PAYLOAD_PRECISION);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " instance-size\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    int size_int = std::stoi(argv[1]);
    InstanceParams prms(static_cast<InstanceSize>(size_int));
    setup_he_context(prms.getSize());

    // Load HE context and public key (raw binary — no zlib overhead)
    auto context = std::make_shared<HEContextImpl<Scheme::CKKS>>(
        load_from_file_raw<HEContextImpl<Scheme::CKKS>>((prms.keydir() / "cc.bin").string()));
    auto pk = load_from_file_raw<Publickey<Scheme::CKKS>>(
        (prms.keydir() / "pk.bin").string());

    HEEncoder<Scheme::CKKS> encoder(context);
    HEEncryptor<Scheme::CKKS> encryptor(context, pk);
    HEArithmeticOperator<Scheme::CKKS> op(context, encoder);

    // 1. Read and transpose Database records
    auto db = read2vecs<float>(prms.datadir() / "db.bin", prms.getRecordDim());
    auto encoded_dataset = transpose_matrix<float>(db, prms.getNSlots());

    // 2. Read and transpose Payloads
    std::vector<std::vector<int16_t>> payloads =
        read2vecs<int16_t>(prms.datadir() / "payloads.bin", PAYLOAD_DIM - 1);
    add_markers(payloads);
    auto encoded_payloads = transpose_matrix<int16_t>(payloads, prms.getNSlots());

    // Scale payloads down by PAYLOAD_PRECISION
    for (auto& batch : encoded_payloads) {
        for (auto& row : batch) {
            for (auto& val : row) val /= PAYLOAD_PRECISION;
        }
    }

    double scale = std::pow(2.0, CKKS_SCALING_MOD_BITS);

    // ---------- 2-level encryption approach (matching reference) ----------
    // DB rows: encrypt at depth = degrees.size()-1 to save space
    int encryption_level1 = static_cast<int>(prms.getDegrees().size()) - 1;
    // Payloads: encrypt at depth 23 (only needed late in pipeline)
    int encryption_level2 = 23;

    // 3. Encrypt and save batches (raw binary, no zlib)
    for (int i = 0; i < prms.getNCtxts(); i++) {
        std::stringstream ssi;
        ssi << std::setw(4) << std::setfill('0') << i;
        auto batch_dir = prms.encdir() / ("batch" + ssi.str());
        fs::create_directories(batch_dir);

        // Encrypt DB rows at encryption_level1
        for (int j = 0; j < prms.getRecordDim(); j++) {
            Plaintext<Scheme::CKKS> pt(context);
            encoder.encode(pt, encoded_dataset[i][j], scale);
            Ciphertext<Scheme::CKKS> ct(context);
            encryptor.encrypt(ct, pt);

            // Mod-drop to the target depth to save space
            for (int d = 0; d < encryption_level1; d++) {
                op.mod_drop_inplace(ct);
            }
#ifdef DEBUG
            if (i == 0 && j == 0) {
                std::cout << "[debug] First row ciphertext depth: " << ct.depth() 
                          << ", level: " << ct.level() << std::endl;
            }
#endif

            std::stringstream ssj;
            ssj << "row_" << std::setw(4) << std::setfill('0') << j << ".bin";
            save_to_file_raw(ct, (batch_dir / ssj.str()).string());
        }

        // Encrypt payloads at encryption_level2, saved as individual files
        for (size_t j = 0; j < PAYLOAD_DIM; j++) {
            Plaintext<Scheme::CKKS> pt(context);
            encoder.encode(pt, encoded_payloads[i][j], scale);
            Ciphertext<Scheme::CKKS> ct(context);
            encryptor.encrypt(ct, pt);

            // Mod-drop to the target depth to save space
            for (int d = 0; d < encryption_level2; d++) {
                op.mod_drop_inplace(ct);
            }
#ifdef DEBUG
            if (i == 0 && j == 0) {
                std::cout << "[debug] First payload ciphertext depth: " << ct.depth() 
                          << ", level: " << ct.level() << std::endl;
            }
#endif
            std::stringstream ssj;
            ssj << "payload_" << std::setw(4) << std::setfill('0') << j << ".bin";
            save_to_file_raw(ct, (batch_dir / ssj.str()).string());
        }
    }
#ifdef DEBUG
    std::cout << "Database successfully encrypted in " << prms.getNCtxts()
              << " batches." << std::endl;
#endif
    return 0;
}
