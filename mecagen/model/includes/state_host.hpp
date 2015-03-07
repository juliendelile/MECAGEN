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

#ifndef _STATE_HOST_
#define _STATE_HOST_

// #include "metaparam.hpp"          
#include "param.hpp"
#include "define.hpp"
#include "state.hpp"

// // Standard :

// // -- I/O related
#include <iostream>
#include <string> //std::getline
#include "stdio.h" //sprintf
#include <fstream>   //ifstream
#include <sstream>   //istringstream


#include <cmath>   //pow
#include <assert.h>

// // -- Collection related
#include <vector>

namespace mg {

  // Forward declarations:
  template<int T> class MetaParam;

  /** State object.
  * This object must be deeply copiable, and follow the rule of three. Here, we use thrust vectors
  * as members so it is indeed the case. However, in the Host-Device mode of the framework, an additional method
  * must be implemented to allow the copy from the backend to the device backend ("copy" method).
  */
  class State_Host: public State<HOST>{

    public:

    /** State class constructor.
    * The thrust vectors are initialized in the constructor initializer list via the MetaParam object passed
    * as a parameter.
    */
    State_Host(MetaParam<HOST>* mp):
          State<HOST>(mp),
          host_only_struct(10)
          {

            randomGaussian_Counter[0] = 0;
            randomUniform_Counter[0] = 0;
            errorCode[0] = 0;
          }

    /** State class destructor. */
    ~State_Host() throw () {}

    /**********************
    /** Host only state ***
    /*********************/

    std::vector<uint>    host_only_struct;
    double               max_c_max;

    /*** - BOOST SERIALIZATION - ***/
    private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
      ar & BOOST_SERIALIZATION_NVP(numCells)
          & BOOST_SERIALIZATION_NVP(currentTimeStep)
          & BOOST_SERIALIZATION_NVP(cellPosition)
          & BOOST_SERIALIZATION_NVP(embryoCenter)
          & BOOST_SERIALIZATION_NVP(cellAxisAB)
          & BOOST_SERIALIZATION_NVP(cellRadius)
          & BOOST_SERIALIZATION_NVP(cellState)
          & BOOST_SERIALIZATION_NVP(cellTimer)
          & BOOST_SERIALIZATION_NVP(cellGeneration)
          & BOOST_SERIALIZATION_NVP(cellCycleLength)
          & BOOST_SERIALIZATION_NVP(cellId_bits)
          & BOOST_SERIALIZATION_NVP(cellLigand)
          & BOOST_SERIALIZATION_NVP(cellEpiIsPolarized)
          & BOOST_SERIALIZATION_NVP(cellProtrusionExtForce)
          & BOOST_SERIALIZATION_NVP(cellMechanoSensorQ)
          & BOOST_SERIALIZATION_NVP(cellProtein)
          & BOOST_SERIALIZATION_NVP(customState)
          ;
    }

  };

}

#endif
