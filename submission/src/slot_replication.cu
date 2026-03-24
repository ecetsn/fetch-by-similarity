#include "slot_replication.cuh"
#include <algorithm>
#include <numeric>

using namespace heongpu;

class ReplicatorNode {
private:
    std::shared_ptr<ReplicatorNode> parent;
    int num_replicas;
    int current;
    int rot_amt;
    std::vector<Ciphertext<Scheme::CKKS>> shifts;
    std::vector<Plaintext<Scheme::CKKS>> masks;
    HEContext<Scheme::CKKS> cc;
    HEOperator<Scheme::CKKS>* op;
    HEEncoder<Scheme::CKKS>* encoder;
    Galoiskey<Scheme::CKKS>* galois_key;

public:
    ReplicatorNode(HEContext<Scheme::CKKS>& _cc, 
                   HEOperator<Scheme::CKKS>& _op,
                   HEEncoder<Scheme::CKKS>& _encoder,
                   Galoiskey<Scheme::CKKS>& _galois_key,
                   std::shared_ptr<ReplicatorNode> _parent, int _nreps, int _amt)
        : cc(_cc), op(&_op), encoder(&_encoder), galois_key(&_galois_key),
          parent(_parent), num_replicas(_nreps), current(_nreps), rot_amt(_amt) {
        shifts.resize(num_replicas);
        generate_masks();
    }

    std::shared_ptr<ReplicatorNode> get_parent() { return parent; }
    int get_num_replicas() { return num_replicas; }

    void generate_masks() {
        int nslots = cc->get_poly_modulus_degree() / 2;
        int block_size = rot_amt * num_replicas;
        int nblocks = nslots / block_size;
        masks.resize(num_replicas, Plaintext<Scheme::CKKS>(cc));
        for (int i = 0; i < num_replicas; i++) {
            std::vector<double> tmp_mask(nslots, 0.0);
            for (int b = 0; b < nblocks; b++) {
                int run_start = b * block_size + i * rot_amt;
                for (int j = 0; j < rot_amt; j++) {
                    tmp_mask[run_start + j] = 1.0;
                }
            }
            encoder->encode(masks[i], tmp_mask, std::pow(2.0, 42.0));
        }
    }

    void install_source(Ciphertext<Scheme::CKKS> ct) {
        shifts[0] = ct;
        for (int i = 1; i < num_replicas; i++) {
            op->rotate_rows(ct, shifts[i], *galois_key, -i * rot_amt);
        }
        current = 0;
    }

    Ciphertext<Scheme::CKKS> init(Ciphertext<Scheme::CKKS> ct) {
        if (parent == nullptr) install_source(ct);
        else install_source(parent->init(ct));
        return next_replica();
    }

    Ciphertext<Scheme::CKKS> next_replica() {
        if (current == num_replicas) {
            if (parent == nullptr) return Ciphertext<Scheme::CKKS>();
            auto next_src = parent->next_replica();
            if (next_src.size() == 0) return Ciphertext<Scheme::CKKS>();
            install_source(next_src);
        }
        
        // Align mask depth to shifts depth before multiply_plain
        // Masks are encoded at depth 0; shifts may be at higher depth from parent levels
        Plaintext<Scheme::CKKS> m0 = masks[current];
        while (m0.depth() < shifts[0].depth()) op->mod_drop_inplace(m0);

        Ciphertext<Scheme::CKKS> acc(cc);
        op->multiply_plain(shifts[0], m0, acc);
        op->rescale_inplace(acc);

        for (int i = 1; i < num_replicas; i++) {
            Plaintext<Scheme::CKKS> mi = masks[(i + current) % num_replicas];
            while (mi.depth() < shifts[i].depth()) op->mod_drop_inplace(mi);

            Ciphertext<Scheme::CKKS> tmp(cc);
            op->multiply_plain(shifts[i], mi, tmp);
            op->rescale_inplace(tmp);
            while (acc.depth() < tmp.depth()) op->mod_drop_inplace(acc);
            while (tmp.depth() < acc.depth()) op->mod_drop_inplace(tmp);
            op->add_inplace(acc, tmp);
        }
        current++;
        return acc;
    }
};

DFSSlotReplicator::DFSSlotReplicator(HEContext<Scheme::CKKS>& context,
                                     HEOperator<Scheme::CKKS>& op,
                                     HEEncoder<Scheme::CKKS>& encoder,
                                     Galoiskey<Scheme::CKKS>& galois_key,
                                     const std::vector<int> tree_degrees,
                                     int input_replication) {
    int num_slots = context->get_poly_modulus_degree() / 2;
    int pattern_len = num_slots / input_replication;
    std::shared_ptr<ReplicatorNode> current = nullptr;
    auto rot_amt = pattern_len;
    for (auto deg : tree_degrees) {
        rot_amt /= deg;
        current = std::make_shared<ReplicatorNode>(context, op, encoder, galois_key, current, deg, rot_amt);
    }
    this->handle = current;
}

Ciphertext<Scheme::CKKS> DFSSlotReplicator::init(Ciphertext<Scheme::CKKS> ct) {
    auto node = std::static_pointer_cast<ReplicatorNode>(handle);
    return node->init(ct);
}

Ciphertext<Scheme::CKKS> DFSSlotReplicator::next_replica() {
    auto node = std::static_pointer_cast<ReplicatorNode>(handle);
    return node->next_replica();
}

std::vector<Ciphertext<Scheme::CKKS>> DFSSlotReplicator::batch_replicate(
    Ciphertext<Scheme::CKKS> ct,
    HEContext<Scheme::CKKS>& context,
    HEOperator<Scheme::CKKS>& op,
    HEEncoder<Scheme::CKKS>& encoder,
    Galoiskey<Scheme::CKKS>& galois_key,
    std::vector<int> tree_degrees, int input_replication) {
    DFSSlotReplicator replicator(context, op, encoder, galois_key, tree_degrees, input_replication);
    std::vector<Ciphertext<Scheme::CKKS>> result;
    for (auto ct_i = replicator.init(ct); ct_i.size() != 0; ct_i = replicator.next_replica()) {
        result.push_back(ct_i);
    }
    return result;
}

std::vector<int> DFSSlotReplicator::get_rotation_amounts(std::vector<int> tree_degrees) {
    std::vector<int> result;
    int total_reps = std::accumulate(tree_degrees.begin(), tree_degrees.end(), 1, std::multiplies<int>());
    // This is a simplification. The actual amounts depend on the pattern length.
    // However, KeyGen usually passes the same degrees.
    // For TOY: pattern_len = 8192, input_replication=64? No.
    // In params.cuh: NSlots=8192, RecordDim=128, input_replication=8192/128=64.
    // pattern_len = 8192 / 64 = 128.
    // degrees = {8, 4, 4}. 8*4*4 = 128. Correct.
    int rot_amt = std::accumulate(tree_degrees.begin(), tree_degrees.end(), 1, std::multiplies<int>());
    for (auto deg : tree_degrees) {
        rot_amt /= deg;
        for (int i = 1; i < deg; i++) {
            result.push_back(-i * rot_amt);
        }
    }
    return result;
}

std::vector<int> DFSSlotReplicator::get_degrees() {
    std::vector<int> result;
    auto current = std::static_pointer_cast<ReplicatorNode>(handle);
    while (current) {
        result.push_back(current->get_num_replicas());
        current = current->get_parent();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

static bool isPowerOfTwo(int n) { return (n > 0) && ((n & (n - 1)) == 0); }

std::vector<int> DFSSlotReplicator::suggest_degrees(int num_outputs) {
    assert(isPowerOfTwo(num_outputs));
    if (num_outputs <= 8) return {num_outputs};
    std::vector<int> degrees;
    degrees.push_back(8);
    num_outputs /= 8;
    if (num_outputs >= 4) {
        degrees.push_back(4);
        num_outputs /= 4;
    }
    while (num_outputs > 1) {
        degrees.push_back(2);
        num_outputs /= 2;
    }
    return degrees;
}
