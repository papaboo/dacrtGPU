// Hyper ray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Material.h>
#include <ToString.h>

#include <iomanip>

std::string Material::ToString() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "[emission: " << emission << ", color: " << color << ", refl: " << reflection << ", refr: " << refraction << "]";
    return out.str();
}
