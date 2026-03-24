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

/**
 * client_encode_encrypt_query:
 * This executable reads the query file, replicates it into FHE slots,
 * and encrypts it using the public key.
 */
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
        heongpu::serializer::load_from_file<HEContextImpl<Scheme::CKKS>>((prms.keydir() / "cc.bin").string()));
    auto pk = heongpu::serializer::load_from_file<Publickey<Scheme::CKKS>>((prms.keydir() / "pk.bin").string());

    HEEncoder<Scheme::CKKS> encoder(context);
    HEEncryptor<Scheme::CKKS> encryptor(context, pk);

    // 1. Read Query
    // Query is expected at [datadir]/query.bin
    auto qs = read2vecs<float>(prms.datadir() / "query.bin", prms.getRecordDim());
    if (qs.empty()) {
        throw std::runtime_error("Query file is empty");
    }
    auto qry = qs[0];

    // 2. Pack Query into slots
    // The query is replicated (n_slots / recordDim) times to fill the slots
    std::vector<double> slots(prms.getNSlots());
    for (int i = 0; i < prms.getNSlots(); i++) {
        slots[i] = static_cast<double>(qry[i % prms.getRecordDim()]);
    }

    // Default scale for CKKS
    double scale = std::pow(2.0, 42); 
    
    // 3. Encrypt Query
    Plaintext<Scheme::CKKS> pt(context);
    encoder.encode(pt, slots, scale);
    Ciphertext<Scheme::CKKS> eqry(context);
    encryptor.encrypt(eqry, pt);
    
    // Save encrypted query to the encrypted directory
    std::filesystem::create_directories(prms.encdir());
    heongpu::serializer::save_to_file(eqry, (prms.encdir() / "query.bin").string());

    std::cout << "Query successfully encrypted and saved." << std::endl;

    return 0;
}
