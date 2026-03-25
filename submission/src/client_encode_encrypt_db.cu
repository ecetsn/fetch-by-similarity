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

    // Load HE context and public key
    auto context = std::make_shared<HEContextImpl<Scheme::CKKS>>(
        load_from_file_raw<HEContextImpl<Scheme::CKKS>>((prms.keydir() / "cc.bin").string()));
    auto pk = load_from_file_raw<Publickey<Scheme::CKKS>>((prms.keydir() / "pk.bin").string());

    HEEncoder<Scheme::CKKS> encoder(context);
    HEEncryptor<Scheme::CKKS> encryptor(context, pk);

    // 1. Read and transpose Database records
    // Database is expected at [datadir]/db.bin
    auto db = read2vecs<float>(prms.datadir() / "db.bin", prms.getRecordDim());
    // Transpose and pack: many records into slots of one ciphertext
    auto encoded_dataset = transpose_matrix<float>(db, prms.getNSlots());

    // 2. Read and transpose Payloads
    // Payloads are expected at [datadir]/payloads.bin
    auto payload_fname = prms.datadir() / "payloads.bin";
    std::vector<std::vector<int16_t>> payloads = read2vecs<int16_t>(payload_fname, PAYLOAD_DIM - 1);
    
    // Add markers and scale payloads as per benchmark specification
    for (auto& p : payloads) {
        // Marker is a high value to identify the start of a payload record
        p.insert(p.begin(), 2 * MAX_PAYLOAD_VAL * PAYLOAD_PRECISION); 
    }
    auto encoded_payloads = transpose_matrix<int16_t>(payloads, prms.getNSlots());
    
    // Normalize payloads (divide by precision)
    for (auto& batch : encoded_payloads) {
        for (auto& row : batch) {
            for (auto& val : row) val /= PAYLOAD_PRECISION;
        }
    }

    // BM Scale
    double scale = std::pow(2.0, 42); 

    // 3. Encrypt and save batches
    for (int i = 0; i < prms.getNCtxts(); i++) {
        std::stringstream ssj;
        ssj << std::setw(4) << std::setfill('0') << i;
        auto batch_dir = prms.encdir() / ("batch" + ssj.str());
        fs::create_directories(batch_dir);

        // Encrypt DB records for this batch, saved as individual row files
        // (row_0000.bin, row_0001.bin, ...) for memory-efficient streaming in mat_vec_mult.
        for (int j = 0; j < prms.getRecordDim(); j++) {
            Plaintext<Scheme::CKKS> pt(context);
            encoder.encode(pt, encoded_dataset[i][j], scale);
            Ciphertext<Scheme::CKKS> ct(context);
            encryptor.encrypt(ct, pt);
            std::stringstream ss;
            ss << "row_" << std::setw(4) << std::setfill('0') << j << ".bin";
            save_to_file_raw(ct, (batch_dir / ss.str()).string());
        }

        if (!prms.isCountOnly()) {
            // Encrypt Payloads for this batch
            std::vector<Ciphertext<Scheme::CKKS>> pay_batch;
            for (size_t j = 0; j < PAYLOAD_DIM; j++) {
                Plaintext<Scheme::CKKS> pt(context);
                encoder.encode(pt, encoded_payloads[i][j], scale);
                Ciphertext<Scheme::CKKS> ct(context);
                encryptor.encrypt(ct, pt);
                pay_batch.push_back(std::move(ct));
            }
            save_batch(pay_batch, (batch_dir / "payloads.bin").string());
        }
    }

    std::cout << "Database successfully encrypted in " << prms.getNCtxts() << " batches." << std::endl;

    return 0;
}
