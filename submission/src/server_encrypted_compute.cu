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

// -----------------------------------------------------------------------------
// Operator Wrapper to expose protected members
// -----------------------------------------------------------------------------

/**
 * HEArithmeticOperator's base class HEOperator has evaluate_poly as protected.
 * This wrapper exposes it for use in the server computation.
 */
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

// -----------------------------------------------------------------------------
// Logging and Utility
// -----------------------------------------------------------------------------

void log_server_step(int num, std::string name, double elapsed = -1.0) {
    auto [timestamp, _] = getCurrentTimeFormatted();
    std::cout << timestamp << " [server] " << num << ": " << name << " completed";
    if (elapsed >= 0) {
        std::cout << " (elapsed " << std::fixed << std::setprecision(0) << elapsed << "s)";
    }
    std::cout << std::endl;
}

static double sigmoid(double x, double outscale = 1.0, double inscale = 69.0) {
    return outscale / (1.0 + std::exp(-(x * inscale)));
}

static double impulse(double x, double sigma = 0.04) {
    double x_over_sigma = x / sigma;
    return std::exp(-x_over_sigma * x_over_sigma / 2);
}

// -----------------------------------------------------------------------------
// Polynomial Evaluation Wrapper
// -----------------------------------------------------------------------------

void evaluate_function_internal(PublicArithmeticOperator& op,
                               Ciphertext<Scheme::CKKS>& ct,
                               const Polynomial& poly,
                               Relinkey<Scheme::CKKS>& relin_key) {
    op.set_poly_type(poly.type_);
    ct = op.evaluate_poly(ct, ct.scale(), poly, relin_key, heongpu::ExecutionOptions());
}

void evaluate_function(PublicArithmeticOperator& op,
                       HEContext<Scheme::CKKS>& cc,
                       HEEncoder<Scheme::CKKS>& encoder,
                       Relinkey<Scheme::CKKS>& relin_key,
                       Ciphertext<Scheme::CKKS>& ct,
                       std::function<double(double)> func,
                       double a, double b, int degree) {

    using Complex = complex_arithmetic::ComplexOperations<double>;
    auto complex_func = [func](Complex x) { return Complex(func(x.real()), 0.0); };
    std::vector<Complex> coeffs = heongpu::approximate_function(complex_func, a, b, degree);

    Polynomial poly(degree, coeffs, false, heongpu::PolyType::CHEBYSHEV, a, b);

    evaluate_function_internal(op, ct, poly, relin_key);
}

// -----------------------------------------------------------------------------
// Core Processing Stages
// -----------------------------------------------------------------------------

std::vector<Ciphertext<Scheme::CKKS>> mat_vec_mult(fs::path encdir,
                                                   Ciphertext<Scheme::CKKS> qry,
                                                   const InstanceParams& prms,
                                                   HEContext<Scheme::CKKS>& cc,
                                                   PublicArithmeticOperator& op,
                                                   HEEncoder<Scheme::CKKS>& encoder,
                                                   Galoiskey<Scheme::CKKS>& galois_key,
                                                   Relinkey<Scheme::CKKS>& relin_key) {

    int input_replication = prms.getNSlots() / prms.getRecordDim();
    DFSSlotReplicator replicator(cc, op, encoder, galois_key, prms.getDegrees(), input_replication);

    int n_batches = prms.getNCtxts();
    std::vector<Ciphertext<Scheme::CKKS>> acc(n_batches);

    // Outer loop: one replica per query dimension (replicator initialized once).
    // Inner loop: one batch at a time — multiply row_ct by replicated query dim, accumulate.
    size_t i = 0;
    for (auto ct_i = replicator.init(qry);
         ct_i.size() != 0 && i < (size_t)prms.getRecordDim();
         ct_i = replicator.next_replica(), i++) {

        std::stringstream ssi;
        ssi << "row_" << std::setw(4) << std::setfill('0') << i << ".bin";

        for (int j = 0; j < n_batches; j++) {
            std::stringstream ssj;
            ssj << std::setw(4) << std::setfill('0') << j;
            Ciphertext<Scheme::CKKS> row_ct(cc);
            load_ciphertext(row_ct, (encdir / ("batch" + ssj.str()) / ssi.str()).string());
            row_ct.store_in_device();
            while (row_ct.depth() < ct_i.depth()) op.mod_drop_inplace(row_ct);

            op.multiply_inplace(row_ct, ct_i);
            op.relinearize_inplace(row_ct, relin_key);
            op.rescale_inplace(row_ct);

            if (i == 0) {
                acc[j] = std::move(row_ct);
            } else {
                while (acc[j].depth() < row_ct.depth()) op.mod_drop_inplace(acc[j]);
                while (row_ct.depth() < acc[j].depth()) op.mod_drop_inplace(row_ct);
                op.add_inplace(acc[j], row_ct);
            }
        }
    }
    return acc;
}

void compare_to_threshold(std::vector<Ciphertext<Scheme::CKKS>>& ctxts,
                          const Polynomial& poly,
                          PublicArithmeticOperator& op,
                          Relinkey<Scheme::CKKS>& relin_key) {
    for (auto& ct : ctxts) {
        evaluate_function_internal(op, ct, poly, relin_key);
    }
}

std::vector<Ciphertext<Scheme::CKKS>> compare_to_number(const std::vector<Ciphertext<Scheme::CKKS>>& ctxts,
                                                        const Polynomial& poly,
                                                        PublicArithmeticOperator& op,
                                                        Relinkey<Scheme::CKKS>& relin_key) {
    std::vector<Ciphertext<Scheme::CKKS>> results;
    results.reserve(ctxts.size());
    for (const auto& ct : ctxts) {
        Ciphertext<Scheme::CKKS> res = ct;
        evaluate_function_internal(op, res, poly, relin_key);
        results.push_back(std::move(res));
    }
    return results;
}

Ciphertext<Scheme::CKKS> total_sums(const Ciphertext<Scheme::CKKS>& ct,
                                    const InstanceParams& prms,
                                    HEContext<Scheme::CKKS>& cc,
                                    PublicArithmeticOperator& op,
                                    Galoiskey<Scheme::CKKS>& galois_key) {
    int period = prms.getNCols() * PAYLOAD_DIM;
    int s = static_cast<int>(std::log2(prms.getNSlots() / period));
    int r = static_cast<int>(std::log2(period));
    Ciphertext<Scheme::CKKS> result = ct;
    // Use POSITIVE rotation amount: HEonGPU positive = left cyclic shift,
    // same convention as OpenFHE EvalRotate with positive argument.
    for (int i = s - 1; i >= 0; i--) {
        int rot_amount = 1 << (i + r);  // positive = left shift
        Ciphertext<Scheme::CKKS> tmp(cc);
        op.rotate_rows(result, tmp, galois_key, rot_amount);
        op.add_inplace(result, tmp);
    }
    return result;
}

// -----------------------------------------------------------------------------
// Main Execution
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        return 0;
    }

    int size_int = std::stoi(argv[1]);
    bool count_only = (argc > 2 && std::string(argv[2]) == "--count_only");
    InstanceParams prms(static_cast<InstanceSize>(size_int), count_only);

    setup_he_context(prms.getSize());

    auto context = std::make_shared<HEContextImpl<Scheme::CKKS>>(
        heongpu::serializer::load_from_file<HEContextImpl<Scheme::CKKS>>((prms.keydir() / "cc.bin").string()));

    auto mk = heongpu::serializer::load_from_file<Relinkey<Scheme::CKKS>>((prms.keydir() / "mk.bin").string());
    mk.store_in_device();

    auto rk = heongpu::serializer::load_from_file<Galoiskey<Scheme::CKKS>>((prms.keydir() / "rk.bin").string());
    rk.store_in_device();

    HEEncoder<Scheme::CKKS> encoder(context);
    PublicArithmeticOperator op(context, encoder);

    log_server_step(0, "Loading keys completed");

    Ciphertext<Scheme::CKKS> eqry(context);
    load_ciphertext(eqry, (prms.encdir() / "query.bin").string());
    eqry.store_in_device();

    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Matrix-vector product
    auto result = mat_vec_mult(prms.encdir(), eqry, prms, context, op, encoder, rk, mk);
    auto t1 = std::chrono::high_resolution_clock::now();
    log_server_step(1, "Matrix-vector product", std::chrono::duration<double>(t1 - start_time).count());

    // Pre-calculate polynomials for efficiency
    using Complex = complex_arithmetic::ComplexOperations<double>;
    auto start_poly = std::chrono::high_resolution_clock::now();
    double threshold = 0.8;
    int sigmoid_degree = count_only ? 247 : 59;

    double outscale = count_only ? 1.0 : 0.504;
    auto sigmoid_func = [threshold, outscale](double x) { return sigmoid(x - threshold, outscale); };
    auto complex_sigmoid = [sigmoid_func](Complex x) { return Complex(sigmoid_func(x.real()), 0.0); };
    std::vector<Complex> sigmoid_coeffs = heongpu::approximate_function(complex_sigmoid, -1.0, 1.0, sigmoid_degree);
    Polynomial sigmoid_poly(sigmoid_degree, sigmoid_coeffs, false, heongpu::PolyType::CHEBYSHEV, -1.0, 1.0);

    // Pre-compute one impulse polynomial per equality-check target x_i = i/4 - 1.
    std::vector<Polynomial> impulse_polys;
    impulse_polys.reserve(prms.getMaxNMatch());
    double impulse_sigma_val = 0.04;
    for (int ii = 1; ii <= prms.getMaxNMatch(); ii++) {
        double tgt = ii / 4.0 - 1.0;  // map i in {1..8} to [-1,1] as per reference
        auto f = [tgt, impulse_sigma_val](Complex x) { return Complex(impulse(x.real() - tgt, impulse_sigma_val), 0.0); };
        std::vector<Complex> coeffs = heongpu::approximate_function(f, -1.0, 1.0, 119);
        impulse_polys.emplace_back(119, coeffs, false, heongpu::PolyType::CHEBYSHEV, -1.0, 1.0);
    }

    auto end_poly = std::chrono::high_resolution_clock::now();
    std::cout << "[server] Polynomial pre-calculation took "
              << std::chrono::duration<double>(end_poly - start_poly).count() << "s" << std::endl;

    // 2. Threshold Comparison
    compare_to_threshold(result, sigmoid_poly, op, mk);
    auto t2 = std::chrono::high_resolution_clock::now();
    log_server_step(2, "Compare to threshold completed", std::chrono::duration<double>(t2 - t1).count());

    if (count_only) {
        log_server_step(3, "Starting server-side summation for count_only");
        // Sum up all ciphertexts in the result vector
        for (size_t i = 1; i < result.size(); i++) {
            while (result[0].depth() < result[i].depth()) op.mod_drop_inplace(result[0]);
            while (result[i].depth() < result[0].depth()) op.mod_drop_inplace(result[i]);
            op.add_inplace(result[0], result[i]);
        }
        // Total sum of slots in result[0]
        int n_slots = context->get_poly_modulus_degree() / 2;
        Ciphertext<Scheme::CKKS> summed = result[0];
        for (int i = 0; i < std::log2(n_slots); i++) {
            Ciphertext<Scheme::CKKS> tmp(context);
            op.rotate_rows(summed, tmp, rk, -(1 << i));
            while (summed.depth() < tmp.depth()) op.mod_drop_inplace(summed);
            while (tmp.depth() < summed.depth()) op.mod_drop_inplace(tmp);
            op.add_inplace(summed, tmp);
        }
        result.clear();
        result.push_back(std::move(summed));

        result[0].store_in_host();
        cudaDeviceSynchronize();
        save_batch(result, (prms.encdir() / "results.bin").string());

        // Write server timing JSON for harness
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto total_s = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            auto comp_s  = std::chrono::duration_cast<std::chrono::seconds>(now - t1).count();
            std::ofstream jf((prms.iodir() / "server_reported_steps.json").string());
            jf << "{\"compute_time_s\": " << comp_s
               << ", \"total_time_s\": " << total_s << "}\n";
        }
    } else {
        // 3. Running Sums
        // Deep-copy result before running sums mutates it.
        std::vector<Ciphertext<Scheme::CKKS>> matches;
        matches.reserve(result.size());
        for (auto& ct : result) {
            Ciphertext<Scheme::CKKS> tmp = ct;  // copy ctor = deep copy
            tmp.store_in_device();
            matches.push_back(std::move(tmp));
        }

        RunningSums rs(context, op, encoder, rk, prms.getNCols(), RUNNING_SUM_LEVELS, result[0].depth());
        rs.eval_in_place(result);

        for (size_t i = 0; i < result.size(); i++) {
            while (matches[i].depth() < result[i].depth()) op.mod_drop_inplace(matches[i]);
            while (result[i].depth() < matches[i].depth()) op.mod_drop_inplace(result[i]);
            op.multiply_inplace(result[i], matches[i]);
            op.relinearize_inplace(result[i], mk);
            op.rescale_inplace(result[i]);
            op.add_plain_inplace(result[i], -1.0);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        log_server_step(3, "Running sums completed", std::chrono::duration<double>(t3 - t2).count());

        // 4. Output Compression
        // For each match index i in [1..MAX_N_MATCH]:
        //   a) compute a one-hot indicator (compare_to_number)
        //   b) multiply each payload column by the indicator, rotating j-th column
        //      by -j*N_COLS to pack them consecutively in the column
        //   c) replicate the packed payload across the whole column (total_sums)
        //   d) mask to keep only positions (i-1)*PAYLOAD_DIM .. i*PAYLOAD_DIM-1 per column
        //   e) accumulate into the final output ciphertext
        Ciphertext<Scheme::CKKS> accumulator(context);
        bool acc_init = false;
        int max_matches = prms.getMaxNMatch();

        for (int i = 1; i <= max_matches; i++) {
            auto indicator = compare_to_number(result, impulse_polys[i - 1], op, mk);

            for (size_t k = 0; k < indicator.size(); k++) {
                // 1. Square to kill side lobes at non-match positions.
                op.multiply_inplace(indicator[k], indicator[k]);
                op.relinearize_inplace(indicator[k], mk);
                op.rescale_inplace(indicator[k]);

                // 2. Clear background noise by masking with the match ciphertext.
                // This eliminates the additive bias (+6) caused by the indicator's noise floor.
                while (matches[k].depth() < indicator[k].depth()) op.mod_drop_inplace(matches[k]);
                while (indicator[k].depth() < matches[k].depth()) op.mod_drop_inplace(indicator[k]);
                op.multiply_inplace(indicator[k], matches[k]);
                op.relinearize_inplace(indicator[k], mk);
                op.rescale_inplace(indicator[k]);

                // 3. Compensate for matches[k] intensity (approx 0.504).
                // Scaling back to approx 1.0.
                op.add_inplace(indicator[k], indicator[k]);
            }


            int n_batches_oc = static_cast<int>(indicator.size());
            Ciphertext<Scheme::CKKS> match_for_idx(context);
            bool match_init = false;

            for (int k = 0; k < n_batches_oc; k++) {
                std::stringstream ssj;
                ssj << std::setw(4) << std::setfill('0') << k;
                auto payloads = load_batch<Scheme::CKKS>(
                    (prms.encdir() / ("batch" + ssj.str()) / "payloads.bin").string(), context);
                for (auto& ct : payloads) ct.store_in_device();

                Ciphertext<Scheme::CKKS> batch_sum(context);
                bool batch_inited = false;
                for (size_t j = 0; j < PAYLOAD_DIM; j++) {
                    Ciphertext<Scheme::CKKS> pay_part = std::move(payloads[j]);
                    while (pay_part.depth() < indicator[k].depth()) op.mod_drop_inplace(pay_part);
                    while (indicator[k].depth() < pay_part.depth()) op.mod_drop_inplace(indicator[k]);
                    op.multiply_inplace(pay_part, indicator[k]);
                    op.relinearize_inplace(pay_part, mk);
                    op.rescale_inplace(pay_part);
                    if (j != 0) op.rotate_rows_inplace(pay_part, rk, -static_cast<int>(j * prms.getNCols()));
                    if (!batch_inited) {
                        batch_sum = std::move(pay_part);
                        batch_inited = true;
                    } else {
                        while (batch_sum.depth() < pay_part.depth()) op.mod_drop_inplace(batch_sum);
                        while (pay_part.depth() < batch_sum.depth()) op.mod_drop_inplace(pay_part);
                        op.add_inplace(batch_sum, pay_part);
                    }
                }

                if (!match_init) {
                    match_for_idx = std::move(batch_sum);
                    match_init = true;
                } else {
                    while (match_for_idx.depth() < batch_sum.depth()) op.mod_drop_inplace(match_for_idx);
                    while (batch_sum.depth() < match_for_idx.depth()) op.mod_drop_inplace(batch_sum);
                    op.add_inplace(match_for_idx, batch_sum);
                }
            }

            if (match_init) {
                // Step c: replicate across every row of each column
                auto replicated = total_sums(match_for_idx, prms, context, op, rk);

                // Step d: plaintext mask — keep only rows (i-1)*PAYLOAD_DIM .. i*PAYLOAD_DIM-1
                // in each column, zero everything else.
                std::vector<double> mask_vec(prms.getNSlots(), 0.0);
                for (int ell = 0; ell < prms.getNSlots(); ell++) {
                    int row = ell / prms.getNCols();  // which row in the column layout
                    if (row >= (i - 1) * PAYLOAD_DIM && row < i * PAYLOAD_DIM) {
                        mask_vec[ell] = 1.0;
                    }
                }
                Plaintext<Scheme::CKKS> mask_pt(context);
                encoder.encode(mask_pt, mask_vec, replicated.scale());
                while (mask_pt.depth() < replicated.depth()) op.mod_drop_inplace(mask_pt);
                Ciphertext<Scheme::CKKS> masked(context);
                op.multiply_plain(replicated, mask_pt, masked);
                op.rescale_inplace(masked);

                // Step e: accumulate
                if (!acc_init) {
                    accumulator = std::move(masked);
                    acc_init = true;
                } else {
                    while (accumulator.depth() < masked.depth()) op.mod_drop_inplace(accumulator);
                    while (masked.depth() < accumulator.depth()) op.mod_drop_inplace(masked);
                    op.add_inplace(accumulator, masked);
                }
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        log_server_step(4, "Output compression completed", std::chrono::duration<double>(t4 - t3).count());

        // Write server timing JSON for harness
        {
            auto total_s = std::chrono::duration_cast<std::chrono::seconds>(t4 - start_time).count();
            auto comp_s  = std::chrono::duration_cast<std::chrono::seconds>(t4 - t1).count();
            std::ofstream jf((prms.iodir() / "server_reported_steps.json").string());
            jf << "{\"compute_time_s\": " << comp_s
               << ", \"total_time_s\": " << total_s << "}\n";
        }

        // Save accumulator using save_batch so client_decrypt_decode can load it via load_batch.
        accumulator.store_in_host();
        cudaDeviceSynchronize();
        std::vector<Ciphertext<Scheme::CKKS>> result_vec;
        result_vec.push_back(std::move(accumulator));
        save_batch(result_vec, (prms.encdir() / "results.bin").string());
    }

    return 0;
}
