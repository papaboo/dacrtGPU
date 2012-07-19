// GPU Dacrt enumerations
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <ostream>

#ifndef _GPU_DACRT_ENUMS_H_
#define _GPU_DACRT_ENUMS_H_

enum SignedAxis { PosX = 0, NegX = 1, PosY = 2, NegY = 3, PosZ = 4, NegZ = 5 };

inline std::ostream& operator<<(std::ostream& s, const SignedAxis a){
    switch(a) {
    case PosX: return s << "+X";
    case NegX: return s << "-X";
    case PosY: return s << "+Y";
    case NegY: return s << "-Y";
    case PosZ: return s << "+Z";
    case NegZ: return s << "-Z";
    default:
        return s << "Unknown";
    }
}

enum Axis { X = 0, Y = 1, Z = 2, U = 3, V = 4 };

inline std::ostream& operator<<(std::ostream& s, const Axis a){
    switch(a) {
    case X: return s << "X";
    case Y: return s << "Y";
    case Z: return s << "Z";
    case U: return s << "U";
    case V: return s << "V";
    default:
        return s << "Unknown";
    }
}

enum PartitionSide { NONE = 0, LEFT = 1, RIGHT = 2, BOTH = 3, 
                     LEFT_LEAF = 4, RIGHT_LEAF = 8, BOTH_LEAVES = 12 };

inline std::ostream& operator<<(std::ostream& s, const PartitionSide ps){
    switch(ps) {
    case NONE: return s << "NONE";
    case LEFT: return s << "LEFT";
    case RIGHT: return s << "RIGHT";
    case BOTH: return s << "BOTH";
    case LEFT_LEAF: return s << "LEFT_LEAF";
    case RIGHT_LEAF: return s << "RIGHT_LEAF";
    case BOTH_LEAVES: return s << "BOTH_LEAVES";
    default:
        return s << "UNKNOWN";
    }
}

struct PartitionSide2 {
    PartitionSide x;
    PartitionSide y;
};

__host__ __device__
inline PartitionSide2 make_PartitionSide2(PartitionSide x,
                                          PartitionSide y) {
    PartitionSide2 res;
    res.x = x; res.y = y;
    return res;
}

inline std::ostream& operator<<(std::ostream& s, const PartitionSide2 ps){
    return s << "[" << ps.x << ", " << ps.y << "]";
}

#endif // _GPU_DACRT_ENUMS_H_
