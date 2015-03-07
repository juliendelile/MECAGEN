/*
 * Copyright (C) 2015 by Julien Delile
 * 
 * This file is part of MECAGEN.
 * 
 * MECAGEN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 3 of the License, or
 * any later version.
 * 
 * MECAGEN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MECAGEN.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "custom.hpp"

namespace mg{

    inline __device__
    double daugtherCellLength(
        const double motherCycleLength,
        const double* randomUniform,
        uint *randomUniform_Counter
        )
    {
      return motherCycleLength * (1 + randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0]) ]);
    }

}
