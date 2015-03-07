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

#ifndef _METAPARAM_H
#define _METAPARAM_H

#include "thrust_objects.hpp"     // Backend<>
#include "define.hpp"     // Backend<>

// Boost -- serialization
#include <boost/serialization/string.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/assume_abstract.hpp>



namespace mg {

  /** MetaParam object.
   * This object must be deeply copiable, and follow the Rule of the Great Three. Here, we use thrust vectors
   * as members so it is indeed the case. In contrast to the Param and State classes, no additional method
   * must be implemented to allow the copy from the backend to the device backend (no "copy" method).
   */
  template<int T>
    class MetaParam {

      public:

        /** MetaParam class constructor.
         * The meta-parameter values are specified here.
         */
        MetaParam():
          numCellsMax(1),
          gridBox_NumPartMax(1),
          grid_SizeMax(1),
          numNeighbMax(1),
          spatialBorderMin(1),
          spatialBorderMax(1),
          devtime_minutes_init(1),
          embryoAxes(3),
          displayScale(1)
      {
        numCellsMax[0]          = NUMCELLmax;  
        gridBox_NumPartMax[0]   = 500;
        grid_SizeMax[0]         = 16;
        numNeighbMax[0]         = NUMNEIGHBMETRICmax;
        spatialBorderMin[0]     = d3(-99999.9,-99999.9,-99999.9);
        spatialBorderMax[0]     = d3(99999.9,99999.9,99999.9);
        devtime_minutes_init[0] = 0;
        embryoAxes[0] = d3(1.0,.0,.0);
        embryoAxes[1] = d3(.0,1.0,.0);
        embryoAxes[2] = d3(.0,.0,1.0);
        displayScale[0] = 1.0;
      }

        /** MetaParam class destructor. */
        ~MetaParam() throw () {}

        /** Is used to store the maximum number of cells.*/
        typename Backend<T>::vecInt       numCellsMax;

        /** Is used to store the maximum number of particles in a grid box.*/
        typename Backend<T>::vecInt       gridBox_NumPartMax;

        /** Is used to store the maximum side length of the cuboidal grid.*/
        typename Backend<T>::vecInt       grid_SizeMax;

        /** Is used to store the maximum number of neighbor ids.*/
        typename Backend<T>::vecInt       numNeighbMax;

        /** Is used to store the spatial coordinate of the simulation world origin.*/
        typename Backend<T>::vecD3       spatialBorderMin;

        /** Is used to store the spatial coordinate of the simulation world limit.*/        
        typename Backend<T>::vecD3       spatialBorderMax;

        /** Is used to store the initial simulation time in minutes.*/
        typename Backend<T>::vecInt      devtime_minutes_init;
       
        /** Is used to store the AP, DV, and left-right axis of the embryo.*/
        typename Backend<T>::vecD3       embryoAxes;

        /** Is used to store the scaling factor determining the original zoom of the GUI display.*/ 
        typename Backend<T>::vecDouble   displayScale;


        /*** - BOOST SERIALIZATION - ***/
      private:
        friend class boost::serialization::access;

        template<class Archive>
          void serialize(Archive & ar, const unsigned int version){
            ar  & BOOST_SERIALIZATION_NVP(numCellsMax)
              & BOOST_SERIALIZATION_NVP(gridBox_NumPartMax)
              & BOOST_SERIALIZATION_NVP(grid_SizeMax)
              & BOOST_SERIALIZATION_NVP(numNeighbMax)
              & BOOST_SERIALIZATION_NVP(spatialBorderMin)
              & BOOST_SERIALIZATION_NVP(spatialBorderMax)
              & BOOST_SERIALIZATION_NVP(devtime_minutes_init)
              & BOOST_SERIALIZATION_NVP(embryoAxes)
              & BOOST_SERIALIZATION_NVP(displayScale)
              ;
          }
    };

} // End namespace

#endif
