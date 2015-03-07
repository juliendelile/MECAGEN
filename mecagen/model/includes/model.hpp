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

#ifndef _MODEL_H_2013_08
#define _MODEL_H_2013_08


#include "thrust_objects.hpp"     //for HOST / DEVICE

// ISF includes
#include "modelproxies.hpp"

namespace mg {
  
  // Forward Declaration:
  class Param_Host;
  class Param_Device;
  class State_Host;
  class State_Device;
  template<int T> class MetaParam;

  /** The Model class.
  * The model is a simple container for two algorithmic methods, "algoStep1" and "algoStep2". In the Host-Device implementation, these
  * methods takes as arguments pointers to the host and device instances of the MetaParam, Param and State objects.
  */
  class Model : public isf::ModelHostDevice<Model, MetaParam<HOST>, Param_Host, Param_Device, State_Host, State_Device >{
    
    public:
      
      /** First part of the user model algorithm implementation. It contains all the code executed before the producers export some data toward
      * the consumers.
      */
      static int algoStep1(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep);
      
      /** Second part of the user model algorithm implementation. It contains all the code executed after the producers export some data toward
      * the consumers. In this case, no code is specified after the producer execution.
      */
      static int algoStep2(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep);
    private:
  };

} // End namespace

#endif
