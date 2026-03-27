// running-sums.cu - Compute running sums acorss ciphertext slots (HEonGPU)
//============================================================================
// Copyright (c) 2025, Amazon Web Services
// All rights reserved.
//
// This software is licensed under the terms of the Apache License v2.
// See the file LICENSE.md for details.
//============================================================================
#include "running_sums.cuh"
#include <cmath>
#include <iostream>

static int divc(int a, int b) {
    return (a + b - 1) / b;
}

static std::vector<double> mask4shift_vec(int amt, int n_slots) {
    amt %= n_slots;
    if (amt < 0) amt += n_slots;
    std::vector<double> mask(n_slots, 0.0);
    for (int i = amt; i < n_slots; i++) {
        mask[i] = 1.0;
    }
    return mask;
}

RunningSums::RunningSums(heongpu::HEContext<heongpu::Scheme::CKKS>& _cc,
                         heongpu::HEOperator<heongpu::Scheme::CKKS>& _op,
                         heongpu::HEEncoder<heongpu::Scheme::CKKS>& _encoder,
                         heongpu::Galoiskey<heongpu::Scheme::CKKS>& _galois_key,
                         int stride, int depth_budget, int level)
    : cc(_cc), op(&_op), encoder(&_encoder), galois_key(&_galois_key) {
    
    int n_slots = cc->get_poly_modulus_degree() / 2;
    int n_intervals = n_slots / stride;
    int logn_intervals = static_cast<int>(std::log2(n_intervals));

    if (depth_budget <= 0 || depth_budget > logn_intervals) {
        depth_budget = logn_intervals;
    }

    int factor = 1 << divc(logn_intervals, depth_budget);

    while (n_intervals > factor) {
        n_intervals /= factor;
        std::map<int, std::vector<double>> phase_masks;
        for (int i = factor - 1; i > 0; i--) {
            int amt = stride * n_intervals * i;
            phase_masks.insert(std::make_pair(-amt, mask4shift_vec(amt, n_slots)));
        }
        mask_slots.push_back(phase_masks);
    }
    if (n_intervals > 1) {
        std::map<int, std::vector<double>> phase_masks;
        for (int i = n_intervals - 1; i > 0; i--) {
            int amt = stride * i;
            phase_masks.insert(std::make_pair(-amt, mask4shift_vec(amt, n_slots)));
        }
        mask_slots.push_back(phase_masks);
    }
}

void RunningSums::eval_in_place(
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& ctxts)
{
    if (ctxts.empty()) return;

    // Step 1: vertical running sums across ciphertexts
    // ctxts[i] <- ctxts[0] + ... + ctxts[i]
    for (size_t i = 1; i < ctxts.size(); i++) {
        while (ctxts[i].depth() < ctxts[i - 1].depth()) {
            op->mod_drop_inplace(ctxts[i]);
        }
        while (ctxts[i - 1].depth() < ctxts[i].depth()) {
            op->mod_drop_inplace(ctxts[i - 1]);
        }
        op->add_inplace(ctxts[i], ctxts[i - 1]);
    }

    // Step 2: per-phase horizontal accumulation from ctxts.back() only
    // Then add the same accumulator to every ciphertext
    for (const auto& phase_masks : mask_slots) {
        bool first_term = true;
        heongpu::Ciphertext<heongpu::Scheme::CKKS> phase_acc(cc);

        for (const auto& [amt, mask_vec] : phase_masks) {
            // Rotate only the last ciphertext 
            heongpu::Ciphertext<heongpu::Scheme::CKKS> rotated(cc);
            op->rotate_rows(ctxts.back(), rotated, *galois_key, amt);

            // Encode mask for this phase/rotation
            heongpu::Plaintext<heongpu::Scheme::CKKS> mask_pt(cc);
            encoder->encode(mask_pt, mask_vec, rotated.scale());

            // Align levels before multiply_plain
            while (mask_pt.depth() < rotated.depth()) {
                op->mod_drop_inplace(mask_pt);
            }
            while (rotated.depth() < mask_pt.depth()) {
                op->mod_drop_inplace(rotated);
            }

            op->multiply_plain_inplace(rotated, mask_pt);
            op->rescale_inplace(rotated);

            if (first_term) {
                phase_acc = std::move(rotated);
                first_term = false;
            } else {
                while (phase_acc.depth() < rotated.depth()) {
                    op->mod_drop_inplace(phase_acc);
                }
                while (rotated.depth() < phase_acc.depth()) {
                    op->mod_drop_inplace(rotated);
                }
                op->add_inplace(phase_acc, rotated);
            }
        }
#ifdef DEBUG
        std::cout << "[debug] RunningSums phase complete, phase_acc depth: " << phase_acc.depth() 
                  << ", level: " << phase_acc.level() << std::endl;
#endif

        if (!first_term) {
            // Make a per-ciphertext copy so each ct gets the same logical accumulator
            for (size_t i = 0; i < ctxts.size(); i++) {
                heongpu::Ciphertext<heongpu::Scheme::CKKS> acc_i = phase_acc;

                while (ctxts[i].depth() < acc_i.depth()) {
                    op->mod_drop_inplace(ctxts[i]);
                }
                while (acc_i.depth() < ctxts[i].depth()) {
                    op->mod_drop_inplace(acc_i);
                }

                op->add_inplace(ctxts[i], acc_i);
            }
        }
    }
}

std::vector<int> RunningSums::get_shift_amounts(int n_slots, int stride, int depth_budget) {
    int logn_intervals = static_cast<int>(std::log2(n_slots / stride));
    if (depth_budget <= 0 || depth_budget > logn_intervals) depth_budget = logn_intervals;
    int factor = 1 << divc(logn_intervals, depth_budget);
    
    std::vector<int> shift_amounts;
    int n_intervals = n_slots / stride;
    while (n_intervals > factor) {
        n_intervals /= factor;
        for (int i = factor - 1; i > 0; i--) {
            shift_amounts.push_back(-stride * n_intervals * i);
        }
    }
    if (n_intervals > 1) {
        for (int i = n_intervals - 1; i > 0; i--) {
            shift_amounts.push_back(-stride * i);
        }
    }
    return shift_amounts;
}

std::vector<std::vector<double>> RunningSums::from_matrix_form(
    const std::vector<std::vector<double>>& matrix, size_t n_slots) {
    if (matrix.empty() || matrix[0].empty()) return {};
    size_t n_rows = matrix.size();
    size_t n_cols = matrix[0].size();
    std::vector<std::vector<double>> slots((n_cols * n_rows) / n_slots, std::vector<double>(n_slots));
    for (size_t i = 0; i < n_rows; i++) {
        size_t slots_i = i % slots.size();
        size_t slots_j = n_cols * (i / slots.size());
        for (size_t j = 0; j < n_cols; j++, slots_j++) {
            slots[slots_i][slots_j] = matrix[i][j];
        }
    }
    return slots;
}

std::vector<std::vector<double>> RunningSums::to_matrix_form(
    const std::vector<std::vector<double>>& slots, size_t n_cols) {
    if (slots.empty() || slots[0].empty()) return {};
    int n_rows_per_vector = slots[0].size() / n_cols;
    std::vector<std::vector<double>> matrix(slots.size() * n_rows_per_vector, std::vector<double>(n_cols));
    for (size_t i = 0; i < matrix.size(); i++) {
        size_t slots_i = i % slots.size();
        size_t slots_j = n_cols * (i / slots.size());
        for (size_t j = 0; j < n_cols; j++, slots_j++) {
            matrix[i][j] = slots[slots_i][slots_j];
        }
    }
    return matrix;
}
