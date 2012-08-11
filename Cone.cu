// Cone
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Cone.h>

#include <iostream>
#include <iomanip>


std::string Cone::ToString() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "[apex: " << apex << ", angle: " << spreadAngle << ", direction: " << direction <<  ", apex distance: " << apexDistance << "]";
    return out.str();
}
