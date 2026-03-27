// server_encrypted_compute.cu - encrypted fetch-by-similarity (HEonGPU)
//============================================================================
// Copyright (c) 2025, Amazon Web Services
// All rights reserved.
//
// This software is licensed under the terms of the Apache License v2.
// See the file LICENSE.md for details.
//============================================================================
#include <cassert>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <memory>
#include <map>
#include <sstream>

#include <cuda_runtime.h>
#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include <heongpu/host/ckks/operator.cuh>

#include "params.cuh"
#include "utils.cuh"
#include "running_sums.cuh"
#include "slot_replication.cuh"

using namespace heongpu;
namespace fs = std::filesystem;

#undef DEBUG

using Clock = std::chrono::steady_clock;

static double elapsed_seconds(const Clock::time_point& start,
                              const Clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

static std::string format_seconds(double s) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << s << "s";
    return oss.str();
}

static void add_breakdown(std::map<std::string, double>& breakdown,
                          const std::string& key, double value) {
    breakdown[key] += value;
}

#ifdef DEBUG
static void printCts(const std::vector<Ciphertext<Scheme::CKKS>>& cts,
                     const std::string& label,
                     std::shared_ptr<HEContextImpl<Scheme::CKKS>> context,
                     HEDecryptor<Scheme::CKKS>& decryptor,
                     HEEncoder<Scheme::CKKS>& encoder) {
    std::cout << label << " [\n";
    for (size_t i = 0; i < cts.size(); i++) {
        if (cts[i].size() == 0) continue;
        Ciphertext<Scheme::CKKS> ct = cts[i];
        ct.store_in_host();
        Plaintext<Scheme::CKKS> pt(context);
        decryptor.decrypt(pt, ct);
        std::vector<double> slots;
        encoder.decode(slots, pt);

        // Find stats: max, min, and positions of large values
        double maxVal = -1e30, minVal = 1e30;
        size_t maxIdx = 0, minIdx = 0;
        int nonzero_count = 0;
        for (size_t j = 0; j < slots.size(); j++) {
            if (slots[j] > maxVal) { maxVal = slots[j]; maxIdx = j; }
            if (slots[j] < minVal) { minVal = slots[j]; minIdx = j; }
            if (std::abs(slots[j]) > 0.01) nonzero_count++;
        }

        std::cout << "  CT " << i << " (depth=" << ct.depth()
                  << ", n_slots=" << slots.size() << "):\n";
        std::cout << "    first 16: [";
        for (size_t j = 0; j < 16 && j < slots.size(); j++) {
            std::cout << std::fixed << std::setprecision(4) << slots[j] << " ";
        }
        std::cout << "]\n";
        std::cout << "    max=" << std::setprecision(4) << maxVal
                  << " @ slot " << maxIdx
                  << ", min=" << minVal << " @ slot " << minIdx
                  << ", nonzero(>0.01)=" << nonzero_count
                  << "/" << slots.size() << "\n";

        // Print all slots with |value| > 0.1 (the "interesting" ones)
        std::cout << "    significant slots (|val|>0.1): ";
        int printed = 0;
        for (size_t j = 0; j < slots.size(); j++) {
            if (std::abs(slots[j]) > 0.1) {
                std::cout << "[" << j << "]=" << std::setprecision(4) << slots[j] << " ";
                if (++printed >= 50) { std::cout << "... (truncated)"; break; }
            }
        }
        if (printed == 0) std::cout << "(none)";
        std::cout << "\n";
    }
    std::cout << "]\n";
}
#endif

// A subclass to expose evaluate_poly publicly for Chebyshev evaluation
class PublicArithmeticOperator : public HEArithmeticOperator<Scheme::CKKS> {
public:
    PublicArithmeticOperator(HEContext<Scheme::CKKS> context, HEEncoder<Scheme::CKKS>& encoder)
        : HEArithmeticOperator<Scheme::CKKS>(context, encoder) {}

    using HEOperator<Scheme::CKKS>::evaluate_poly;
    using Polynomial = HEOperator<Scheme::CKKS>::Polynomial;

    void set_poly_type(heongpu::PolyType type) {
        this->eval_mod_config_.poly_type_ = type;
    }
};

using Polynomial = PublicArithmeticOperator::Polynomial;

// A utility function to get one encrypted ciphertext from the dataset. This
// implementation assumes that ciphertexts are just separate files on disk,
// it should be re-written if they are streamed from a remote location.
inline Ciphertext<Scheme::CKKS> get_ctxt(HEContext<Scheme::CKKS>& cc,
                                          fs::path ct_name) {
    Ciphertext<Scheme::CKKS> ct(cc);
    load_ciphertext(ct, ct_name.string());
    ct.store_in_device();
    return ct;
}

// Matrix-vector product: The matrix rows are stored on disk in batches
// under iodir/<size>/encrypted/batchNNNN/. The query ciphertext contains
// the query vector, repeated to fill in all the slots.
std::vector<Ciphertext<Scheme::CKKS>> mat_vec_mult(fs::path encdir,
                Ciphertext<Scheme::CKKS> qry, const InstanceParams& prms,
                HEContext<Scheme::CKKS>& cc,
                PublicArithmeticOperator& op,
                HEEncoder<Scheme::CKKS>& encoder,
                Galoiskey<Scheme::CKKS>& galois_key,
                Relinkey<Scheme::CKKS>& relin_key,
                double& io_time_acc,
                double& he_time_acc,
                std::map<std::string, double>& breakdown);

// Compare each slot in the ctxts to the threshold, using a Chebyshev
// approximation of the indicator function chi(x) = (x >= threshold).
// Rather than approximating 0/1 outcome, we scale it to 0/0.5, since we
// will sum up upto eight matches, then multiply by the original thing,
// and need to fit the result to a size-2 interval that can be shifted
// to [+-1].
void compare_to_threshold(std::vector<Ciphertext<Scheme::CKKS>>& ctxts,
                          double threshold, bool count_only,
                          PublicArithmeticOperator& op,
                          Relinkey<Scheme::CKKS>& relin_key);

// Compare each slot in the ciphertexts to the number, using a Chebyshev
// approximation of the function chi(x) = (x == number).
std::vector<Ciphertext<Scheme::CKKS>> compare_to_number(
    const std::vector<Ciphertext<Scheme::CKKS>>& ctxts, double number,
    PublicArithmeticOperator& op,
    Relinkey<Scheme::CKKS>& relin_key);

// Read from disk the ith payload value of all the records, namely the
// i'th row of the payload matrix.
Ciphertext<Scheme::CKKS> get_encrypted_payload(HEContext<Scheme::CKKS>& cc,
                                                fs::path datadir, size_t batch,
                                                size_t idx);

// A SIMD-optimized procedure for computing total sums. The slots are viewed
// as a matrix, and total sums are computed in each column separately.
// All the entries of an output column contain the total sum of entries from
// that column in the input.
Ciphertext<Scheme::CKKS> total_sums(
    const Ciphertext<Scheme::CKKS>& ct, const InstanceParams& prms,
    HEContext<Scheme::CKKS>& cc,
    PublicArithmeticOperator& op,
    Galoiskey<Scheme::CKKS>& galois_key);

/*******************************************************************/
int main(int argc, char* argv[]) {
    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    bool count_only = (argc > 2 && std::string(argv[2]) == "--count_only");

    InstanceParams prms(size, count_only);
    constexpr double threshold = 0.8;
    auto timing_fname = prms.iodir() / "server_reported_steps.json";
    auto start_server = Clock::now();

    double io_time = 0.0;
    double he_time = 0.0;
    std::map<std::string, double> breakdown;

    // Setup / loading is not encrypted computation, so report it as I/O/setup.
    {
        auto t_io = Clock::now();

        setup_he_context(prms.getSize());

        // Read the crypto context and the keys from disk
        auto context = std::make_shared<HEContextImpl<Scheme::CKKS>>(
            load_from_file_raw<HEContextImpl<Scheme::CKKS>>((prms.keydir() / "cc.bin").string()));

        auto mk = load_from_file_raw<Relinkey<Scheme::CKKS>>((prms.keydir() / "mk.bin").string());
        mk.store_in_device();

        auto rk = load_from_file_raw<Galoiskey<Scheme::CKKS>>((prms.keydir() / "rk.bin").string());
        rk.store_in_device();

#ifdef DEBUG // Read also the secret key for debugging
        auto sk = load_from_file_raw<Secretkey<Scheme::CKKS>>((prms.keydir() / "sk.bin").string());
#endif

        HEEncoder<Scheme::CKKS> encoder(context);
        PublicArithmeticOperator op(context, encoder);

#ifdef DEBUG
        HEDecryptor<Scheme::CKKS> decryptor(context, sk);
#endif

        // Read the query vector from disk
        Ciphertext<Scheme::CKKS> eqry(context);
        load_ciphertext(eqry, (prms.encdir() / "query.bin").string());
        eqry.store_in_device();

        double dt = elapsed_seconds(t_io, Clock::now());
        io_time += dt;
        add_breakdown(breakdown, "Setup and loading", dt);

#ifdef DEBUG
        std::cout << "[debug] Loaded query ciphertext depth: " << eqry.depth()
                  << ", level: " << eqry.level() << std::endl;
#endif

        // Matrix-vector multiplication, reading the encrypted matrix one
        // ciphertext at a time from encdir
        auto result = mat_vec_mult(prms.encdir(), eqry, prms, context, op, encoder, rk, mk,
                                   io_time, he_time, breakdown);

#ifdef DEBUG
        if (!result.empty()) {
            std::cout << "[debug] Result[0] depth after mat-vec mult: " << result[0].depth()
                      << ", level: " << result[0].level() << std::endl;
            printCts(result, "After Mat-Vec Mult:", context, decryptor, encoder);
        }
#endif

        // Compare each slot in the results ctxts to the threshold, using a
        // Chebyshev approximation of the indicator function chi(x)=(x>=threshold).
        // If we only want to count the matches, then we use a higher-degree
        // approximation since (a) we care about good approximation for both matches
        // and non-matches and (b) we can afford it level-wise.
        // Otherwise we use lower-degree approximation since we care a little less
        // about the accuracy of matches, more about non-matches (as we have more of
        // them). Also, we scale it to 0/0.5 rather than 0/1, since we sum up upto
        // eight matches, then multiply by the original thing, and need to fit the
        // result to a size-2 interval that can be shifted to the interval [-1,1].
#ifdef DEBUG
        if (!result.empty()) {
            std::cout << "[debug] Before threshold comparison: depth=" << result[0].depth()
                      << ", level=" << result[0].level() << std::endl;
        }
#endif
        {
            auto t_he = Clock::now();
            compare_to_threshold(result, threshold, count_only, op, mk);
            cudaDeviceSynchronize();
            double dt_he = elapsed_seconds(t_he, Clock::now());
            he_time += dt_he;
            add_breakdown(breakdown, "Compare to threshold", dt_he);
        }
#ifdef DEBUG
        if (!result.empty()) {
            std::cout << "[debug] After threshold comparison: depth=" << result[0].depth()
                      << ", level=" << result[0].level() << std::endl;
            printCts(result, " match vector:", context, decryptor, encoder);
        }
#endif

        // If we only want to count matches, return the total sum
        // of all the slots in all the ciphertexts.
        if (count_only) {
            auto t_he = Clock::now();

            for (size_t i = 1; i < result.size(); i++) {
                while (result[0].depth() < result[i].depth()) op.mod_drop_inplace(result[0]);
                while (result[i].depth() < result[0].depth()) op.mod_drop_inplace(result[i]);
                op.add_inplace(result[0], result[i]);
            }
            // Total sum of all slots
            int n_slots = prms.getNSlots();
            Ciphertext<Scheme::CKKS> summed = result[0];
            for (int i = 0; i < static_cast<int>(std::log2(n_slots)); i++) {
                Ciphertext<Scheme::CKKS> tmp(context);
                op.rotate_rows(summed, tmp, rk, -(1 << i));
                while (summed.depth() < tmp.depth()) op.mod_drop_inplace(summed);
                while (tmp.depth() < summed.depth()) op.mod_drop_inplace(tmp);
                op.add_inplace(summed, tmp);
            }

            cudaDeviceSynchronize();
            double dt_he = elapsed_seconds(t_he, Clock::now());
            he_time += dt_he;
            add_breakdown(breakdown, "Summation", dt_he);

            result.clear();
            result.push_back(std::move(summed));
#ifdef DEBUG
            printCts(result, " summed match vector:", context, decryptor, encoder);
#endif

            // Store the result back to disk
            {
                auto t_io2 = Clock::now();
                result[0].store_in_host();
                cudaDeviceSynchronize();
                save_batch(result, (prms.encdir() / "results.bin").string());
                double dt_io2 = elapsed_seconds(t_io2, Clock::now());
                io_time += dt_io2;
                add_breakdown(breakdown, "Result write", dt_io2);
            }

            double total_time = elapsed_seconds(start_server, Clock::now());
            {
                std::ofstream jf(timing_fname.string());
                jf << "{\n";
                jf << "  \"Server Reported\": {\n";
                jf << "    \"Encrypted computation\": \"" << format_seconds(he_time) << "\",\n";
                jf << "    \"I/O\": \"" << format_seconds(io_time) << "\",\n";
                jf << "    \"Total\": \"" << format_seconds(total_time) << "\",\n";
                jf << "    \"Breakdown\": {\n";
                bool first = true;
                for (const auto& kv : breakdown) {
                    if (!first) jf << ",\n";
                    jf << "      \"" << kv.first << "\": \"" << format_seconds(kv.second) << "\"";
                    first = false;
                }
                jf << "\n";
                jf << "    }\n";
                jf << "  }\n";
                jf << "}\n";
            }
            return 0;
        }

        // Make a deep copy of the matches, it will be multiplied back into the
        // result after the running-sum procedure
        std::vector<Ciphertext<Scheme::CKKS>> matches;
        matches.reserve(result.size());
        for (auto& ct : result) {
            Ciphertext<Scheme::CKKS> tmp = ct;  // copy ctor = deep copy
            tmp.store_in_device();
            matches.push_back(std::move(tmp));
        }

        // The "compaction" procedure views the matches vector (made of multiple
        // ciphertexts of dimension N_SLOTS) as a matrix with N_COLS=prms.getNCols()
        // columns, and expect no more than eight matches per column. The columns
        // are packed equally-spaced in the ciphertexts, so each ciphertext contains
        // N_SLOTS/N_COLS entries from each column.

        // Running sums in each column, so the first match will have value 1,
        // the second match will have 2, etc.
        {
            auto t_he = Clock::now();

            RunningSums rs(context, op, encoder, rk, prms.getNCols(), RUNNING_SUM_LEVELS, result[0].depth());
            rs.eval_in_place(result);  // The actual running-sums procedure

#ifdef DEBUG
            printCts(result, " after running sums:", context, decryptor, encoder);
#endif

            // Multiply by the matches vector, to zero out all the non-matches
            for (size_t i = 0; i < result.size(); i++) {
                while (matches[i].depth() < result[i].depth()) op.mod_drop_inplace(matches[i]);
                while (result[i].depth() < matches[i].depth()) op.mod_drop_inplace(result[i]);
                op.multiply_inplace(result[i], matches[i]);
                op.relinearize_inplace(result[i], mk);
                op.rescale_inplace(result[i]);
            }
            // Note: matches are NOT cleared here (unlike the reference), because
            // they are reused below to mask the indicator and reduce noise.

            // Contents of slots are now in the range [0,2], shift them to [-1,1]
            for (auto& ct : result) {
                op.add_plain_inplace(ct, -1.0);
            }

            cudaDeviceSynchronize();
            double dt_he = elapsed_seconds(t_he, Clock::now());
            he_time += dt_he;
            add_breakdown(breakdown, "Running sums", dt_he);
        }

        // We now get the actual payload data corresponding to the matches. Recall
        // that we expect at most MAX_N_MATCH(=8) matches per column: the 1st is
        // marked by a 1 slot in the result ciphertext, the 2nd by a 2 slot, etc.
        // Recall also that we have PAYLOAD_DIM(=8) of payload slots per record.

        // To get the actual data, we run MAX_N_MATCH(=8) iterations, in the i'th
        // iteration we isolate the PAYLOAD_DIM payload slots of the ith match
        // (i.e., the slot that contains i). We first compute an "one hot" indicator
        // ctxt with 1 in the slots where result has i, and zero elsewhere (so we
        // have a single 1 per column).

        // Once we compute the i'th indicator, we need to extract the PAYLOAD_DIM
        // payload entries in the columns corresponding to the 1s in this indicator,
        // then move them slots {i*PAYLOAD_DIM,...,(i+1)*PAYLOAD_DIM-1} in their
        // column. We do it in four steps:
        // 1. We multiply each of the PAYLOAD_DIM encrypted payload vectors by the
        //   indicator vector. This yields PAYLOAD_DIM vectors with the jth one
        //   containing the jth payload value of the records corresponding to the
        //   1s in the indicator. Each column has at most one non-zero payload
        //   values, all in the same slot index.
        // 2. We tile these PAYLOAD_DIM vectors so that the non-zero values appear
        //   in consecutive positions in the column. Since columns are spread
        //   across the slots then it means that the PAYLOAD_DIM payload slots
        //   for one record will appear in slots {x, x+N_COLS, x+2*N_COLS,...},
        //   where x is the slot where the indicator has 1 (in that column).
        // 3. We replicate the values across that column, so that it contains
        //   these PAYLOAD_DIM values repeatedly in all the slots in that column.
        // 4. We multiply the result by a mask which is 1 in positions
        //   {i*PAYLOAD_DIM,...,(i+1)*PAYLOAD_DIM-1} in each column and zero
        //   elsewhere.

        Ciphertext<Scheme::CKKS> accumulator(context);
        bool acc_init = false;
        for (int i = 1; i <= prms.getMaxNMatch(); i++) {  // extract i'th match
            double x_i = i / 4.0 - 1.0;  // map from {0,8} to the interval [-1,1]

#ifdef DEBUG
            if (i == 1) printCts(result, "result before compare_to_number, i=1:", context, decryptor, encoder);
#endif

            std::vector<Ciphertext<Scheme::CKKS>> indicator;
            {
                auto t_he = Clock::now();
                indicator = compare_to_number(result, x_i, op, mk);
                cudaDeviceSynchronize();
                double dt_he = elapsed_seconds(t_he, Clock::now());
                he_time += dt_he;
                add_breakdown(breakdown, "Compare to number", dt_he);
            }

            // Indicator has as many ciphertexts as it takes to store a row of the keys
            // matrix (i.e., one slot for each dataset record). It's a "one hot" vector
            // per column, containing 1 in slots where partial_sums contained i

#ifdef DEBUG
            if (i == 1) printCts(indicator, "Indicator for match 1 index:", context, decryptor, encoder);
#endif

            {
                auto t_he = Clock::now();
                for (size_t k = 0; k < indicator.size(); k++) {
                    // 1. Square to kill side lobes at non-match positions
                    op.multiply_inplace(indicator[k], indicator[k]);
                    op.relinearize_inplace(indicator[k], mk);
                    op.rescale_inplace(indicator[k]);

                    // 2. Clear background noise by masking with the match ciphertext.
                    //    This eliminates the additive bias caused by the indicator's
                    //    noise floor.
                    while (matches[k].depth() < indicator[k].depth()) op.mod_drop_inplace(matches[k]);
                    while (indicator[k].depth() < matches[k].depth()) op.mod_drop_inplace(indicator[k]);
                    op.multiply_inplace(indicator[k], matches[k]);
                    op.relinearize_inplace(indicator[k], mk);
                    op.rescale_inplace(indicator[k]);

                    // 3. Compensate for matches[k] intensity (approx 0.504).
                    //    Scaling back to approx 1.0.
                    op.add_inplace(indicator[k], indicator[k]);
                }
                cudaDeviceSynchronize();
                double dt_he = elapsed_seconds(t_he, Clock::now());
                he_time += dt_he;
                add_breakdown(breakdown, "Indicator sharpening", dt_he);
            }

#ifdef DEBUG
            if (i == 1) printCts(indicator, "Indicator for match 1 (after sharpening):", context, decryptor, encoder);
#endif

            // A place holder for the extracted payload, before moving them to
            // their place in the output columns.
            Ciphertext<Scheme::CKKS> to_replicate(context);
            for (size_t j = 0; j < PAYLOAD_DIM; j++) {
                // Steps 1 & 2: Multiply by the indicator to get a single payload value
                // per column, then rotate by j*N_COLS to put that value in the next
                // available slot in its column.
                for (size_t k = 0; k < indicator.size(); k++) {
                    auto t_io_payload = Clock::now();
                    auto payload_part = get_encrypted_payload(context, prms.encdir(), k, j);
                    double dt_io_payload = elapsed_seconds(t_io_payload, Clock::now());
                    io_time += dt_io_payload;
                    add_breakdown(breakdown, "Payload extraction (I/O)", dt_io_payload);
                    // jth row in the k'th matrix

#ifdef DEBUG
                    if (k == 0 && j == 0 && i == 1) {
                        std::cout << "[debug] Payload depth: " << payload_part.depth()
                                  << ", level: " << payload_part.level()
                                  << ", Indicator depth: " << indicator[k].depth()
                                  << ", Indicator level: " << indicator[k].level() << std::endl;
                    }
#endif

                    auto t_he_payload = Clock::now();

                    while (payload_part.depth() < indicator[k].depth()) op.mod_drop_inplace(payload_part);
                    while (indicator[k].depth() < payload_part.depth()) op.mod_drop_inplace(indicator[k]);
                    op.multiply_inplace(payload_part, indicator[k]);
                    op.relinearize_inplace(payload_part, mk);
                    op.rescale_inplace(payload_part);

                    // Shift the j'th payload value by j positions in its column, so we
                    // pack all PAYLOAD_DIM=8 values consecutively in their column.
                    if (j == 0 && k == 0) {   // initialize the inner-loop accumulator
                        to_replicate = payload_part;  // deep copy via assignment
                    } else {
                        if (j != 0) {  // shift by j in its column
                            op.rotate_rows_inplace(payload_part, rk, -static_cast<int>(j * prms.getNCols()));
                        }
                        while (to_replicate.depth() < payload_part.depth()) op.mod_drop_inplace(to_replicate);
                        while (payload_part.depth() < to_replicate.depth()) op.mod_drop_inplace(payload_part);
                        op.add_inplace(to_replicate, payload_part);  // accumulate
                    }
                    // ? Note : When the small dataset is executed with seed 12345, this assumption does not hold.
                    // Meaning that : Multiple i values (matches) are contributing to the same column block so
                    // indicators overlape across different i values : block = payload_r1 + payload_r2 + ... + payload_rn
                    // To overcome this issue, we need to sharpen the indicator by taking the squae
                    // or multiply the indicator by matches[k]

                    // We assume that indicator has a single 1 in each output column and
                    // all else are zero. So for each slot index s<N_SLOTS, at most one
                    // of the values added to to_replicate[s] will be non-zero. This lets
                    // us use a single ciphertext for to_replicate, even though the
                    // indicator is a vector of ciphertexts, we just add everything and
                    // are assured that at most one of the terms is non-zero.

                    cudaDeviceSynchronize();
                    double dt_he_payload = elapsed_seconds(t_he_payload, Clock::now());
                    he_time += dt_he_payload;
                    add_breakdown(breakdown, "Payload extraction (HE)", dt_he_payload);
                }
            }

#ifdef DEBUG
            if (i == 1) printCts({to_replicate}, "to_replicate (before total_sums):", context, decryptor, encoder);
#endif

            // Step 3: replicate the values across the column
            // We need to move the (potential) PAYLOAD_DIM non-zero slots in each
            // output column to positions {i*PAYLOAD_DIM,...,(i+1)*PAYLOAD_DIM-1}
            // in that column. This is done by first replicating them so that they
            // fill the entire column, then multiplying by a mask that zeros out
            // everything else, leaving only those positions.
            Ciphertext<Scheme::CKKS> replicated(context);
            {
                auto t_he = Clock::now();
                replicated = total_sums(to_replicate, prms, context, op, rk);
                cudaDeviceSynchronize();
                double dt_he = elapsed_seconds(t_he, Clock::now());
                he_time += dt_he;
                add_breakdown(breakdown, "Total sums", dt_he);
            }

#ifdef DEBUG
            if (i == 1) printCts({replicated}, "replicated (after total_sums):", context, decryptor, encoder);
#endif

            // Step 4: multiply by a mask
            {
                auto t_he = Clock::now();

                std::vector<double> slots(prms.getNSlots(), 0.0);
                for (size_t ell = 0; ell < slots.size(); ell++) {
                    int row = ell / prms.getNCols();  // index within column
                    if (row >= (i - 1) * PAYLOAD_DIM && row < i * PAYLOAD_DIM) {
                        slots[ell] = 1.0;
                    }
                }
                Plaintext<Scheme::CKKS> mask(context);
                encoder.encode(mask, slots, replicated.scale());
                while (mask.depth() < replicated.depth()) op.mod_drop_inplace(mask);
                Ciphertext<Scheme::CKKS> masked(context);
                op.multiply_plain(replicated, mask, masked);
                op.rescale_inplace(masked);

                // Finally, add the payload values to all the other matches in that column
                if (!acc_init) {  // initialize the outer accumulator
                    accumulator = std::move(masked);
                    acc_init = true;
                } else {
                    while (accumulator.depth() < masked.depth()) op.mod_drop_inplace(accumulator);
                    while (masked.depth() < accumulator.depth()) op.mod_drop_inplace(masked);
                    op.add_inplace(accumulator, masked);
                }

                cudaDeviceSynchronize();
                double dt_he = elapsed_seconds(t_he, Clock::now());
                he_time += dt_he;
                add_breakdown(breakdown, "Final masking and accumulation", dt_he);
            }
        }
        matches.clear();          // not needed anymore
        matches.shrink_to_fit();  // release the memory

        // Store the accumulated result back to disk
        {
            auto t_io2 = Clock::now();
            accumulator.store_in_host();
            cudaDeviceSynchronize();
            std::vector<Ciphertext<Scheme::CKKS>> result_vec;
            result_vec.push_back(std::move(accumulator));
            save_batch(result_vec, (prms.encdir() / "results.bin").string());
#ifdef DEBUG
            printCts(result_vec, "Final Accumulator:", context, decryptor, encoder);
#endif
            double dt_io2 = elapsed_seconds(t_io2, Clock::now());
            io_time += dt_io2;
            add_breakdown(breakdown, "Result write", dt_io2);
        }

        double total_time = elapsed_seconds(start_server, Clock::now());
        {
            auto fmt = [](double s) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << s << "s";
                return oss.str();
            };

            std::ofstream jf(timing_fname.string());
            jf << "{\n";
            jf << "  \"Server Reported\": {\n";
            jf << "    \"Encrypted computation\": \"" << fmt(he_time) << "\",\n";
            jf << "    \"I/O\": \"" << fmt(io_time) << "\",\n";
            jf << "    \"Total\": \"" << fmt(total_time) << "\",\n";
            jf << "    \"Breakdown\": {\n";

            std::vector<std::string> ordered_keys = {
                "Setup and loading",
                "Slot replication setup",
                "Matrix-vector product (I/O)",
                "Matrix-vector product (HE)",
                "Compare to threshold",
                "Running sums",
                "Summation",
                "Compare to number",
                "Indicator sharpening",
                "Payload extraction (I/O)",
                "Payload extraction (HE)",
                "Total sums",
                "Final masking and accumulation",
                "Result write"
            };

            bool first = true;
            for (const auto& key : ordered_keys) {
                auto it = breakdown.find(key);
                if (it == breakdown.end()) continue;
                if (!first) jf << ",\n";
                jf << "      \"" << key << "\": \"" << fmt(it->second) << "\"";
                first = false;
            }

            jf << "\n";
            jf << "    }\n";
            jf << "  }\n";
            jf << "}\n";
        }
    }

    return 0;
}
/*******************************************************************/
/*******************************************************************/


/*******************************************************************/
// Matrix-vector product: The matrix rows are stored on disk in batches
// under iodir/<size>/encrypted/batchNNNN/. The query ciphertext contains
// the query vector, repeated N_SLOTS/RECORD_DIM many times to fill all
// the slots.
std::vector<Ciphertext<Scheme::CKKS>> mat_vec_mult(fs::path encdir,
                Ciphertext<Scheme::CKKS> qry, const InstanceParams& prms,
                HEContext<Scheme::CKKS>& cc,
                PublicArithmeticOperator& op,
                HEEncoder<Scheme::CKKS>& encoder,
                Galoiskey<Scheme::CKKS>& galois_key,
                Relinkey<Scheme::CKKS>& relin_key,
                double& io_time_acc,
                double& he_time_acc,
                std::map<std::string, double>& breakdown)
{
    // The input ciphertext includes a pattern of length RECORD_DIM,
    // repeated N_SLOTS/RECORD_DIM many times to fill all the slots
    auto n_reps = prms.getNSlots() / prms.getRecordDim();

    auto t_he_rep = Clock::now();
    DFSSlotReplicator replicator(cc, op, encoder, galois_key, prms.getDegrees(), n_reps);
    cudaDeviceSynchronize();
    double dt_he_rep = elapsed_seconds(t_he_rep, Clock::now());
    he_time_acc += dt_he_rep;
    add_breakdown(breakdown, "Slot replication setup", dt_he_rep);

    auto n_batches = prms.getNCtxts();
    std::vector<Ciphertext<Scheme::CKKS>> acc(n_batches);  // an accumulator
    size_t i = 0;  // i is the ciphertext index within a batch
    for (auto ct_i = replicator.init(qry); ct_i.size() != 0;
         ct_i = replicator.next_replica(), i++) {
        // ct_i has the i'th entry of the query vector in all its slots

        // read a row from each batch, multiply by ct_i and accumulate
        std::stringstream ssi;
        ssi << "row_" << std::setw(4) << std::setfill('0') << i << ".bin";
        for (int j = 0; j < n_batches; j++) {  // j is the batch index
            std::stringstream ssj;
            ssj << std::setw(4) << std::setfill('0') << j;

            auto ct_fname = prms.encdir() /
                ("batch" + ssj.str()) / ssi.str();

            auto t_io = Clock::now();
            Ciphertext<Scheme::CKKS> ct = get_ctxt(cc, ct_fname);
            double dt_io = elapsed_seconds(t_io, Clock::now());
            io_time_acc += dt_io;
            add_breakdown(breakdown, "Matrix-vector product (I/O)", dt_io);

            auto t_he = Clock::now();

            // In HEonGPU, we must align depths before multiply (unlike OpenFHE
            // which does EvalMultNoRelin without explicit depth matching).
            while (ct.depth() < ct_i.depth()) op.mod_drop_inplace(ct);

            op.multiply_inplace(ct, ct_i);
            // Note: We must relinearize after each multiply because HEonGPU
            // cannot add 3-poly ciphertexts to 2-poly ones. The reference
            // defers relinearization to after all accumulations.
            op.relinearize_inplace(ct, relin_key);
            op.rescale_inplace(ct);

            if (i == 0) {  // initialize the accumulator
                acc[j] = std::move(ct);
            } else {       // add to the accumulator
                while (acc[j].depth() < ct.depth()) op.mod_drop_inplace(acc[j]);
                while (ct.depth() < acc[j].depth()) op.mod_drop_inplace(ct);
                op.add_inplace(acc[j], ct);
            }

            cudaDeviceSynchronize();
            double dt_he = elapsed_seconds(t_he, Clock::now());
            he_time_acc += dt_he;
            add_breakdown(breakdown, "Matrix-vector product (HE)", dt_he);
        }
#ifdef DEBUG
        if (i % 32 == 0 || i == (size_t)prms.getRecordDim() - 1) {
            std::cout << "[debug] mat_vec_mult: processed row " << i
                      << ", current acc[0] depth: " << acc[0].depth()
                      << ", level: " << acc[0].level() << std::endl;
        }
#endif
    }
    return acc;
}

/*******************************************************************/
// Compare each slot in the results ctxts to the threshold, using a
// Chebyshev approximation of the indicator function chi(x)=(x>=threshold).
// If we only want to count the matches, then we use a higher-degree
// approximation since (a) we care about good approximation for both matches
// and non-matches and (b) we can afford it level-wise.
// Otherwise we use lower-degree approximation since we care a little less
// about the accuracy of matches, more about non-matches (as we have more of
// them). Also, we scale it to 0/0.5 rather than 0/1, since we sum up upto
// eight matches, then multiply by the original thing, and need to fit the
// result to a size-2 interval that can be shifted to [+-1].

// A sigmoid-like function. The constant 69 was determined by experiments
constexpr double sigmoid_inscale = 69.0;
static double sigmoid(double x, double outscale = 1.0,
                      double inscale = sigmoid_inscale) {
    return outscale / (1.0 + std::exp(-(x * inscale)));
}

void compare_to_threshold(std::vector<Ciphertext<Scheme::CKKS>>& ctxts,
                          double threshold, bool count_only,
                          PublicArithmeticOperator& op,
                          Relinkey<Scheme::CKKS>& relin_key) {
    double outscale = count_only ? 1.0 : 0.504;
    auto func = [threshold, outscale](double x) {
        return sigmoid(x - threshold, outscale);
    };
    // In fetch mode, the desired output for matches is ~0.504 (not 1.0). Using a
    // a higher degree gives a more accurate approximation, so matched
    // values are mapped close to this target scale.
    //
    // With a lower degree (59), the approximation is less accurate: it may not
    // reach the intended outscale (~0.504) reliably, and the offset on non-matching
    // values is no longer uniform. Because of that, a single fixed correction term
    // cannot be applied safely in the degree-59 case.
    size_t degree = (count_only ? 247 : 119);

    // Pre-compute the polynomial coefficients once
    using Complex = complex_arithmetic::ComplexOperations<double>;
    auto complex_func = [func](Complex x) { return Complex(func(x.real()), 0.0); };
    std::vector<Complex> coeffs = heongpu::approximate_function(complex_func, -1.0, 1.0, degree);
    Polynomial poly(degree, coeffs, false, heongpu::PolyType::CHEBYSHEV, -1.0, 1.0);

    for (auto& ct : ctxts) {
        op.set_poly_type(poly.type_);
        ct = op.evaluate_poly(ct, ct.scale(), poly, relin_key, heongpu::ExecutionOptions());
    }
    // Benchmark Reference Note: "If these results are not accurate enough then we can either switch
    // to higher-degree approximation or just square the result to get a better
    // approximation of the non-matches."
}

/*******************************************************************/
// Compare each point in the vectors to the number, using a Chebyshev
// approximation of the function chi(x) = (x == number).

// An impulse-like function, with impulse(0)==1.
// The constant 0.04 was determined by experiments.
constexpr double impulse_sigma = 0.04;
static double impulse(double x, double sigma = impulse_sigma) {
    double x_over_sigma = x / sigma;
    return std::exp(-x_over_sigma * x_over_sigma / 2);
}

std::vector<Ciphertext<Scheme::CKKS>> compare_to_number(
    const std::vector<Ciphertext<Scheme::CKKS>>& ctxts, double number,
    PublicArithmeticOperator& op,
    Relinkey<Scheme::CKKS>& relin_key) {
    auto func = [number](double x) {
        return impulse(x - number);
    };
    constexpr size_t degree = 119;  // options are 59, 119, 247

    // Pre-compute the polynomial coefficients once
    using Complex = complex_arithmetic::ComplexOperations<double>;
    auto complex_func = [func](Complex x) { return Complex(func(x.real()), 0.0); };
    std::vector<Complex> coeffs = heongpu::approximate_function(complex_func, -1.0, 1.0, degree);
    Polynomial poly(degree, coeffs, false, heongpu::PolyType::CHEBYSHEV, -1.0, 1.0);

    std::vector<Ciphertext<Scheme::CKKS>> results;
    results.reserve(ctxts.size());
    for (const auto& ct : ctxts) {
        Ciphertext<Scheme::CKKS> res = ct;
        op.set_poly_type(poly.type_);
        res = op.evaluate_poly(res, res.scale(), poly, relin_key, heongpu::ExecutionOptions());
        results.push_back(std::move(res));
    }
    return results;
}

/*******************************************************************/
// A SIMD-optimized procedure for computing total sums. The slots are viewed
// as a matrix, and total sums are computed in each column separately.
// All the entries of an output column contain the total sum of entries from
// that column in the input.
Ciphertext<Scheme::CKKS> total_sums(
    const Ciphertext<Scheme::CKKS>& ct, const InstanceParams& prms,
    HEContext<Scheme::CKKS>& cc,
    PublicArithmeticOperator& op,
    Galoiskey<Scheme::CKKS>& galois_key) {
    int period = prms.getNCols() * PAYLOAD_DIM;
    int s = std::log2(prms.getNSlots() / period);
    int r = std::log2(period);
    assert(unsigned(prms.getNSlots()) == 1UL << (s + r));  // must be a power of two
    Ciphertext<Scheme::CKKS> results = ct;

    // Total sums inside the vectors, in columns
    for (int i = s - 1; i >= 0; i--) {
        // cyclic rotation of results by 2^{i+r}
        int rot_amount = 1 << (i + r);
        Ciphertext<Scheme::CKKS> tmp(cc);
        op.rotate_rows(results, tmp, galois_key, rot_amount);
        // Add tmp back to the results
        op.add_inplace(results, tmp);
    }
    return results;
}

// Read the ith payload value in a batch of records from disk
Ciphertext<Scheme::CKKS> get_encrypted_payload(HEContext<Scheme::CKKS>& cc,
                                                fs::path datadir, size_t batch,
                                                size_t idx) {
    std::stringstream ssi, ssj;
    ssj << std::setw(4) << std::setfill('0') << batch;
    ssi << std::setw(4) << std::setfill('0') << idx;
    auto dir = datadir / ("batch" + ssj.str());
    auto ct_fname = dir / ("payload_" + ssi.str() + ".bin");

    // read the i'th payload ciphertext from this batch
    return get_ctxt(cc, ct_fname);
}