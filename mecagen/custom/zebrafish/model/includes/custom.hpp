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

#ifndef _CUSTOM_
#define _CUSTOM_

#include "custom_objects.hpp"

namespace mg{
  
  class Param_Host;
  class Param_Device;
  class State_Host;
  class State_Device;
  template<int T> class MetaParam;

  inline __device__
  double daugtherCellLength(
    const double motherCycleLength,
    const double* randomUniform,
    uint *randomUniform_Counter );
  
  void custom_algo_neighb(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep);
  void custom_algo_forces(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep, uint loop);
  void custom_algo_evl_growth_division(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep);
  void custom_algo_yolk_evl_diffusion(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep);
}

#define CUSTOM_CELL_CYCLE           daugtherCellLength(cellcyclelength, randomUniform, randomUniform_Counter);

#define CUSTOM_REGEL_FUNCTION_2     0;

#define CUSTOM_ALGO_NEIGHB      custom_algo_neighb(mp, p, pd, s, sd, currentTimeStep);
#define CUSTOM_ALGO_FORCES      custom_algo_forces(mp, p, pd, s, sd, currentTimeStep, loop);
#define CUSTOM_ALGO_MITOSIS     custom_algo_evl_growth_division(mp, p, pd, s, sd, currentTimeStep);
#define CUSTOM_ALGO_REGULATION  custom_algo_yolk_evl_diffusion(mp, p, pd, s, sd, currentTimeStep);

#endif
