#include "utils.cuh"
#include <heongpu/util/memorypool.cuh>
#include <iostream>
#include <cmath>

void setup_he_context(InstanceSize size) {
    cudaFree(nullptr);
    heongpu::MemoryPoolConfig config = heongpu::MemoryPoolConfig::Defaults();
    config.initial_device_fraction = 0.4f;
    config.max_device_fraction = 0.9f;
    config.initial_host_fraction = 0.1f;
    config.max_host_fraction = 0.5f;
    heongpu::MemoryPool::instance().initialize(config);
}

std::vector<int> build_q_bits(int depth) {
    std::vector<int> bits(depth, CKKS_SCALING_MOD_BITS);
    bits.insert(bits.begin(), CKKS_FIRST_MOD_BITS);
    return bits;
}
