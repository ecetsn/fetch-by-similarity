#ifndef PARAMS_H_
#define PARAMS_H_

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

constexpr int RUNNING_SUM_LEVELS = 3;

// The payload slots contain numbers in the range [0,MAX_PAYLOAD_VAL]
// with precision of 1/PAYLOAD_PRECISION
constexpr int MAX_PAYLOAD_VAL = 256;
constexpr int PAYLOAD_PRECISION = 16;

// The dimension of the payload vectors (currently fixed to 8)
constexpr int PAYLOAD_DIM = 8;

// an enum for benchmark size
enum InstanceSize {
    TOY = 0,
    SMALL = 1,
    MEDIUM = 2,
    LARGE = 3
};

inline std::string instance_name(const InstanceSize size) {
    if (unsigned(size) > unsigned(InstanceSize::LARGE)) {
        return "unknown";
    }
    static const std::string names[] = {"toy", "small", "medium", "large"};
    return names[int(size)];
}

// Parameters that differ for different instance sizes
class InstanceParams {
    InstanceSize size;
    bool count_only;
    int recordDim;  // dimension of the plaintext record
    int dbSize;     // number of records in the dataset
    int ringDim;    // HE ring dimension
    std::vector<int> degrees;  // must multiply to the record dimension
    fs::path rootdir; // root of the submission dir structure (see below)

public:
    // Constructor
    explicit InstanceParams(InstanceSize _size, bool _count_only = false,
                            fs::path _rootdir = fs::current_path())
                            : size(_size), count_only(_count_only), rootdir(_rootdir)
    {
        if (unsigned(_size) > unsigned(InstanceSize::LARGE)) {
            throw std::invalid_argument("Invalid instance size");
        }
        // Parameters for sizes:       toy   small   medium      large
        static const int recDims[] = { 128,   128,     256,      512};
        static const int dbSizes[] = {1000, 50000, 1000000, 20000000};
        
        ringDim = (_size == InstanceSize::TOY)? 4096 : 65536;
        recordDim = recDims[int(_size)];
        dbSize    = dbSizes[int(_size)];

        // NOTE: The degrees vector specifies the shape of the tree used by
        // by the slot replicator. The entires must multiply to the record
        // dimension, and for a given shape the slot-replicator consumes
        // degrees.size() levels of mult-by-constant.
        // In theory, given a depth bound d, the best shape of the tree should
        // have been {dim/2^{d-1}, 2, ..., 2}, but in practice this is not
        // what happens. Maybe due to multi-threading??
        // Below are some fixed shapes for the different sizes. These are
        // unlikely to be optimal, the optimal shape is likely dependent on
        // the specific hardware platform. But at least for the larger sizes,
        // the replication time should be insignificant.
        switch (_size) {
            case InstanceSize::LARGE:
                degrees = {16, 8, 4};
                break;
            case InstanceSize::MEDIUM:
                degrees = {8, 8, 4};
                break;
            default:
                degrees = {8, 4, 4};
        }
    }

    InstanceSize getSize() const { return size; }
    bool isCountOnly() const { return count_only; }
    int getRecordDim() const { return recordDim; }
    int getDbSize() const { return dbSize; }
    std::vector<int> getDegrees() const { return degrees; }
    int getNSlots() const { return ringDim/2; } 

    int getNCtxts() const {
        return (dbSize + getNSlots() - 1) / getNSlots();
    }
    int getRingDim() const { return ringDim; }
    int getNCols() const { return ringDim/128; }
    int getMaxNMatch() const { return 64 / PAYLOAD_DIM; };

    fs::path rtdir() const  { return rootdir; }
    // Harness uses [root]/io/toy/keys, NOT [root]/toy/keys
    fs::path iodir() const  { return rootdir / "io" / instance_name(size); }
    fs::path keydir() const { return iodir() / "keys"; }
    fs::path encdir() const { return iodir() / "encrypted"; }
    fs::path datadir() const {
        return rootdir/"datasets"/instance_name(size);
    }
};

#endif  // ifdef PARAMS_H_
