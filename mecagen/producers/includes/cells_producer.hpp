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

#ifndef _CELLSPRODUCER_HPP_2013_11
#define _CELLSPRODUCER_HPP_2013_11

// Mother class:
#include "producer.hpp"

// Demo class:
#include "metaparam.hpp"
#include "state.hpp"
#include "thrust_objects.hpp"

// Standard 
#include <cstring>

namespace mg{

  // Header of the structure exported by this producer
  typedef struct EmbryoState{
    uint    numCells;
    uint    gridSize;
    double  gridBoxSize;
    double  worldOrigin;
    uint    TS;
    // CustomState customState;
    CustomStateBuffer customStateBuffer;
    d3*     cellPosition;
    d3*     cellAxis1;
    d3*     cellAxisAB;
    d3*     cellRadius;
    uint*   cellMetricNeighbId;
    uint*   cellMetricNeighbNum;
    uint*   cellTopologicalNeighbId;
    uint*   cellTopologicalNeighbNum;
    uint*   cellPopulation;
    uint*   cellId_bits;
    uint*   cellTimer;
    uint*   cellGeneration;
    double* cellLigand;
    uint*   cellType;
    uint*   cellEpiIsPolarized;
    uint*   cellEpiId;
    d3*     cellCandidateAxes;
    double* cellProtein;
  } embryoState;

  /** CellsProducer, example of client-specified Producer_Impl_Host_Device class. 
  * Here, this producer is associated with the ForestVisualizer2D consumer, i.e. set up as
  * a producer parameter in the ForestVisualizer2D constructor. As the CellsProducer is
  * used in the Host-Device context, it is template with both the host and device versions of the
  * MetaParam and State types.
  */
  class CellsProducer:
    public isf::Producer_Impl_Host_Device<MetaParam<HOST>, State<HOST>, State<DEVICE> > {

      public: 
        /** CellsProducer class constructor. */
        CellsProducer(){}

        /** CellsProducer class destructor. */
        virtual ~CellsProducer() throw (){};

      protected:
        /** Implements the behavior of the producer when it is executed during an iteration of the simulation loop. 
        * It takes as arguments:
        * @param[in] s      A pointer toward the host instance of the State object
        * @param[in] sd     A pointer toward the device instance of the State object
        * @param[in] step   The time step of the simulation at which the producer is executed
        * @param[in] buffer A pointer toward the associated broker's buffer, on which the exported data are written
        */
        void doProcessing(State<HOST> *s, State<DEVICE> *sd, int step, void * buffer);
        
        /** Returns the size of the required buffer. It is used to initialize the associated broker.*/
        size_t getRequestedBufferSize();
    };

  }

#endif

