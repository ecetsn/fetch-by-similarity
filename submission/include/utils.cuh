#ifndef FHEBENCH_UTILS_H_
#define FHEBENCH_UTILS_H_
// utils.h - Utility declerations for fetch-by-similarity
//============================================================================
// Copyright (c) 2025, Amazon Web Services
// All rights reserved.
//
// This software is licensed under the terms of the Apache License v2.
// See the file LICENSE.md for details.
//============================================================================
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

template<typename T>
std::vector<T> vector_union(std::vector<std::vector<T> >& vecs)
{
  std::set<T> combinedSet;

  // Insert elements from both vectors into the set
  for (const auto& v : vecs) {
    combinedSet.insert(v.begin(), v.end());
  }

  // Create a new vector from the set
  std::vector<T> result(combinedSet.begin(), combinedSet.end());
  return result;
}

/// Read a binary file into a vector of vectors, all of dimension record_dim
template<typename T> std::vector<std::vector<T>> read2vecs(
    std::filesystem::path fname, int record_dim)
{
  std::ifstream file(fname, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open " + fname.string() + " for read");
  }
  // Calculate size of the matrix
  file.seekg(0, std::ios::end);
  std::streampos nbytes = file.tellg();
  file.seekg(0, std::ios::beg);
  auto nrecords = nbytes / (record_dim * sizeof(T));

  std::vector<std::vector<T>> a(nrecords);
  for (auto& r : a) {
    r.resize(record_dim);
    file.read(reinterpret_cast<char*>(&r[0]), record_dim * sizeof(T));
  }
  file.close();
  return a;
}

// Write a binary file containing the matrix in vecs
template<typename T> void write2disk(
    std::filesystem::path fname,const std::vector<std::vector<T>>& vecs)
{
  std::ofstream file(fname, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open " + fname.string() + " for write");
  }
  for (auto& v : vecs) {
    file.write(reinterpret_cast<const char*>(&v[0]), v.size() * sizeof(T));
  }
  file.close();
}

/// Encode the dataset in column order: The input is an n-by-m matrix that
/// we want to transpose, but the rows of the output cannot have dimension
/// above n_slots. To accomodate input matrices with more than n_slots rows,
/// the output is split into ceil(n/n_slots) matrices, each of dimension
/// m-by-n_slots, where the rows of the last one may be padded with zeros.
template<typename T>
std::vector<std::vector<std::vector<double> > > transpose_matrix(
    std::vector<std::vector<T> > &mat, size_t n_slots)
{
  // ceil( mat.size()/n_slots )
  auto n_ctxt_per_row = (mat.size() + n_slots - 1) / n_slots;
  auto record_dim = mat[0].size();

  // Allocate space
  std::vector<std::vector<std::vector<double>>> transposed(n_ctxt_per_row);
  for (auto& batch : transposed) {
    batch.resize(record_dim);
    for (auto& record : batch) {
      record.assign(n_slots, 0.0);
    }
  }

  // encode in batches of n_slots records at a time
  for (size_t i = 0; i < n_ctxt_per_row; i++) {  // go over the batches
    // transpose the next n_slots rows in db
    for (size_t j = 0; j < record_dim; j++) {
      for (size_t k = 0; k < n_slots; k++) {
        auto idx = (i * n_slots) + k;
        if (idx < mat.size()) {
          transposed[i][j][k] = mat[idx][j];
        } else {
          break;
        }
      }
    }
  }
  return transposed;  // return the encoded matrix
}

#include <chrono>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <heongpu/heongpu.hpp>
#include "params.cuh"

/// Returns the current time in the format H:M:S, and also duration
/// since last call in seconds (or 0 if this is the first call).
inline std::pair<std::string, double> getCurrentTimeFormatted() {
    using namespace std::chrono;
    static system_clock::time_point previous;
    auto now = system_clock::now();
    auto now_c = std::time_t(system_clock::to_time_t(now));

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%H:%M:%S");

    double duration = -1.0;
    if (previous != system_clock::time_point{}) {
        duration = duration_cast<milliseconds>(now - previous).count() / 1000.0;
    }
    previous = now;
    return {ss.str(), duration};
}

// HEonGPU specific utilities
void setup_he_context(InstanceSize size);
void configure_memory_pool(float initial_fraction = 0.3f, float max_fraction = 0.9f);

/**
 * @brief Serialize an object to a raw byte buffer (no compression).
 */
template <typename T>
std::vector<uint8_t> serialize_raw(const T& obj) {
    std::stringstream ss;
    obj.save(ss);
    return heongpu::serializer::to_buffer(ss);
}

/**
 * @brief Deserialize an object from a raw byte buffer (no decompression).
 */
template <typename T>
void deserialize_raw(T& obj, const std::vector<uint8_t>& buffer) {
    std::stringstream ss;
    heongpu::serializer::from_buffer(ss, buffer);
    obj.load(ss);
}

/**
 * @brief Save a serializable object to a raw binary file.
 */
template <typename T>
void save_to_file_raw(const T& obj, const std::string& filename) {
    auto data = serialize_raw(obj);
    uint64_t size = data.size();
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open " + filename + " for writing");
    ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char*>(data.data()), size);
}

/**
 * @brief Load a serializable object from a raw binary file.
 */
template <typename T>
T load_from_file_raw(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open " + filename + " for reading");
    uint64_t size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::vector<uint8_t> buffer(size);
    ifs.read(reinterpret_cast<char*>(buffer.data()), size);
    
    T obj;
    deserialize_raw(obj, buffer);
    return obj;
}

template <heongpu::Scheme SchemeType>
void save_batch(const std::vector<heongpu::Ciphertext<SchemeType>>& batch, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) throw std::runtime_error("Cannot open " + filename + " for write");
    uint64_t num_elements = static_cast<uint64_t>(batch.size());
    ofs.write(reinterpret_cast<const char*>(&num_elements), sizeof(num_elements));
    for (const auto& ct : batch) {
        auto data = serialize_raw(ct);
        uint64_t size = static_cast<uint64_t>(data.size());
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
        ofs.write(reinterpret_cast<const char*>(data.data()), size);
    }
}

template <heongpu::Scheme SchemeType>
void load_ciphertext(heongpu::Ciphertext<SchemeType>& ct, const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open " + filename + " for read");
    uint64_t size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::vector<uint8_t> buffer(size);
    ifs.read(reinterpret_cast<char*>(buffer.data()), size);
    
    deserialize_raw(ct, buffer);
}

template <heongpu::Scheme SchemeType>
std::vector<heongpu::Ciphertext<SchemeType>> load_batch(const std::string& filename, std::shared_ptr<heongpu::HEContextImpl<SchemeType>> context) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open " + filename + " for read");
    uint64_t num_elements;
    ifs.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
    std::vector<heongpu::Ciphertext<SchemeType>> batch;
    batch.reserve(num_elements);
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::vector<uint8_t> buffer(size);
        ifs.read(reinterpret_cast<char*>(buffer.data()), size);
        
        heongpu::Ciphertext<SchemeType> ct(context);
        deserialize_raw(ct, buffer);
        batch.push_back(std::move(ct));
    }
    return batch;

  }

#endif  // ifdef FHEBENCH_UTILS_H_
