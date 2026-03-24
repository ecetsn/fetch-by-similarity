#ifndef PARAMS_H_
#define PARAMS_H_

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

// The level budget for the running-sums procedure
constexpr int RUNNING_SUM_LEVELS = 3;

// CKKS modulus chain parameters
constexpr int CKKS_SCALING_MOD_BITS = 42;    // scaling-modulus size
constexpr int CKKS_FIRST_MOD_BITS = 57;      // first-modulus size

// The payload slots contain numbers in the range [0,MAX_PAYLOAD_VAL]
// with precision of 1/PAYLOAD_PRECISION
constexpr int MAX_PAYLOAD_VAL = 256;
constexpr int PAYLOAD_PRECISION = 16;

// The dimension of the payload vectors (currently fixed to 8)
constexpr int PAYLOAD_DIM = 8;

// Special primes used for Hybrid KeySwitching (appended to coeff modulus)
inline std::vector<int> get_special_primes_bits() { return {60, 60, 60}; }

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
    int ringDim;    // dimenion of the FHE ring
    int multDepth;  // multiplicative depth
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
        
        recordDim = recDims[int(_size)];
        dbSize    = dbSizes[int(_size)];

        if (_size == InstanceSize::TOY) {
            ringDim = 16384;
            multDepth = 25;
        } else {
            ringDim = 65536;
            multDepth = 25;
        }

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
    int getRingDim() const { return ringDim; }
    int getMultDepth() const { return multDepth; }
    std::vector<int> getDegrees() const { return degrees; }
    int getNSlots() const { return ringDim/2; } 

    int getNCtxts() const {
        return (dbSize + getNSlots() - 1) / getNSlots();
    }

    int getNCols() const { return (size == InstanceSize::TOY) ? 128 : 512; }
    int getMaxNMatch() const { return 8; };

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
