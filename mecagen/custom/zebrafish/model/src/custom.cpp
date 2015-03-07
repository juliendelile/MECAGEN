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

#include "param_host.hpp"
#include "param_device.hpp"
#include "state_host.hpp"
#include "state_device.hpp"

#include "spatialNeighborhood.hpp"
#include "customSpatialNeighborhood.hpp"
#include "customPolarization.hpp"
#include "customForces.hpp"
#include "customEvlDivision.hpp"
#include "customDiffusion.hpp"

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

    void custom_algo_neighb(
              MetaParam<HOST>* mp, 
              Param_Host * p, 
              Param_Device * pd, 
              State_Host * s, 
              State_Device * sd, 
              int currentTimeStep)
    {

      /********************************************/
      /********************************************/
      /*******  YOLK internal neighborhood   ******/
      /********************************************/
      /********************************************/

      thrust::counting_iterator<int> first_yolkPart(0);     
      thrust::counting_iterator<int> first_yolkInteriorPart = first_yolkPart + NUMPARTYOLKMEMBRANE;     
      thrust::counting_iterator<int> last_yolkPart = first_yolkPart + NUMPARTYOLK;

      // // yolk boundary is defined by the its membrane, i.e. we suppose all yolk interior particle
      // // remain inside the membrane.
      d3 firstyolkpart = sd->customState.yolkPosition[0];
      boundingBox init_yolk = boundingBox(firstyolkpart,firstyolkpart);
      bbox_reduction binary_op_yolk;
      boundingBox bbox_yolk = thrust::reduce(
                      (*sd).customState.yolkPosition.begin(),
                      (*sd).customState.yolkPosition.begin() + NUMPARTYOLK,
                      // cptr,
                      // cptr + NUMPARTYOLKMEMBRANE,
                      init_yolk,
                      binary_op_yolk);
      double yolkworldMax = std::max(std::max(bbox_yolk.upper_right.x, bbox_yolk.upper_right.y), bbox_yolk.upper_right.z);
      double yolkworldMin = std::min(std::min(bbox_yolk.lower_left.x, bbox_yolk.lower_left.y), bbox_yolk.lower_left.z);
      double yolkworldSize = yolkworldMax - yolkworldMin;
      double yolkGridBoxSize = s->customState.yolkInteriorDistMax[0];
      int yolkGridSize = std::floor(yolkworldSize / yolkGridBoxSize) + 1; 

      if(yolkGridSize >= YOLKGRIDSIZEmax){
          std::cout << "the yolk size ("<<yolkGridSize
                    <<") is larger than the maximum size allowed. Increase metaparameter's grid_SizeMax ("<< YOLKGRIDSIZEmax <<") to allow larger yolk." << std::endl;
          sd->errorCode[0] = 101;
      }

      // empty grid box cells counter
      thrust::fill_n( 
                (*sd).customState.yolkGridPartNum.begin(),
                yolkGridSize * yolkGridSize * yolkGridSize, 
                0);

      thrust::for_each(
          first_yolkPart,
          last_yolkPart,
          fill_grid(
              thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),  
              yolkGridBoxSize,
              yolkworldMin,
              yolkGridSize,
              thrust::raw_pointer_cast(&((*sd).customState.yolkGridPartNum[0])),
              thrust::raw_pointer_cast(&((*sd).customState.yolkGridPartId[0])),
              YOLKGRIDNUMPARTmax,
              thrust::raw_pointer_cast(&((*sd).errorCode[0])),
              currentTimeStep
          )
      );

      thrust::for_each(
          first_yolkInteriorPart,
          last_yolkPart,
          custom_yolk_metric_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            yolkGridBoxSize,
            yolkworldMin,
            yolkGridSize,
            thrust::raw_pointer_cast(&((*sd).customState.yolkGridPartNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkGridPartId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      thrust::for_each(
          first_yolkInteriorPart,
          last_yolkPart,
          custom_yolk_topological_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /*********  Yolk-Cells neighborhood ********/
      /*******************************************/
      /*******************************************/

      double cellsyolkworldMax = std::max(s->worldMax[0], yolkworldMax);
      double cellsyolkworldMin = std::min(s->worldOrigin[0], yolkworldMin);
      double cellsyolkworldSize = cellsyolkworldMax - cellsyolkworldMin;

      //deduce cell maximum radius from main gridbox size
      double cellRadmax = s->gridBoxSize[0] / (2.0 * s->max_c_max);
      double ymRadius = s->customState.yolkMembraneRadius[0];
      double cellsyolkGridBoxSize = p->customParam[0].cellsYolkCmax * (cellRadmax + ymRadius);

      int cellsyolkGridSize = std::floor(cellsyolkworldSize / cellsyolkGridBoxSize) + 1; 
      if(cellsyolkGridSize >= CELLSYOLKGRIDSIZEmax){
          std::cout << "the cells+yolk size ("<< cellsyolkGridSize
                    <<") is larger than the maximum size allowed. Increase metaparameter's grid_SizeMax ("<< CELLSYOLKGRIDSIZEmax <<") to allow larger yolk." << std::endl;
          sd->errorCode[0] = 101;
      }


      
      thrust::fill_n( 
                (*sd).customState.cellsYolkGridPartNum.begin(),
                cellsyolkGridSize * cellsyolkGridSize * cellsyolkGridSize, 
                0);

      thrust::counting_iterator<int> first_cell(0);       
      thrust::counting_iterator<int> last_cell = first_cell + s->numCells[0];

      thrust::for_each(
          first_cell,
          last_cell,
          fill_grid(
              thrust::raw_pointer_cast(&((*sd).cellPosition[0])),  
              cellsyolkGridBoxSize,
              cellsyolkworldMin,
              cellsyolkGridSize,
              thrust::raw_pointer_cast(&((*sd).customState.cellsYolkGridPartNum[0])),
              thrust::raw_pointer_cast(&((*sd).customState.cellsYolkGridPartId[0])),
              CELLSYOLKGRIDNUMPARTmax,
              thrust::raw_pointer_cast(&((*sd).errorCode[0])),
              currentTimeStep
          )
      );


      thrust::fill_n( 
                (*sd).customState.cellsYolkNeighbNum.begin(),
                NUMCUSTOMCELLmax, 
                0);

      thrust::for_each(
          first_yolkPart,
          first_yolkInteriorPart,
          custom_cellsyolk_metric_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            p->customParam[0].cellsYolkCmax,
            ymRadius,
            cellsyolkGridBoxSize,
            cellsyolkworldMin,
            cellsyolkGridSize,
            thrust::raw_pointer_cast(&((*sd).customState.cellsYolkGridPartNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.cellsYolkGridPartId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneActivated[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkCellsNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkCellsNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.cellsYolkNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /*******  EVL internal neighborhood ********/
      /*******************************************/
      /*******************************************/

      // Get current numEVL from device
      uint numEVLCells = sd->customState.numPartEVL[0];
      
      // Get device customState members addresses
      thrust::counting_iterator<int> first_evlPart(0);     
      thrust::counting_iterator<int> last_evlPart = first_evlPart + numEVLCells;

      d3 firstevlpart = sd->customState.evlPosition[0];
      boundingBox init_evl = boundingBox(firstevlpart,firstevlpart);
      bbox_reduction binary_op_evl;
      boundingBox bbox_evl = thrust::reduce(
                      (*sd).customState.evlPosition.begin(),
                      (*sd).customState.evlPosition.begin() + numEVLCells,
                      init_evl,
                      binary_op_evl);
      double evlworldMax = std::max(std::max(bbox_evl.upper_right.x, bbox_evl.upper_right.y), bbox_evl.upper_right.z);
      double evlworldMin = std::min(std::min(bbox_evl.lower_left.x, bbox_evl.lower_left.y), bbox_evl.lower_left.z);
      // Evl world size is increased to also contain the cells (see cells-EVL neighborhood)
      evlworldMax = std::max(evlworldMax, s->worldMax[0]);
      evlworldMin = std::min(evlworldMin, s->worldOrigin[0]);
      // Evl world size is also increased to contain the yolk membrane (see Yolk margin-EVL neighborhood)
      evlworldMax = std::max(evlworldMax, yolkworldMax);
      evlworldMin = std::min(evlworldMin, yolkworldMin);
      
      double evlworldSize = evlworldMax - evlworldMin;

      thrust::device_vector<d3>::iterator radMax = thrust::max_element(
                            (*sd).customState.evlRadius.begin(),
                            (*sd).customState.evlRadius.begin() + numEVLCells,
                            compare_first_radius_value());
      d3 radmax = *radMax;

      double evlGridBoxSize = 2.0 * p->customParam[0].evlCmax * radmax.x;

      int evlGridSize = std::floor(evlworldSize / evlGridBoxSize) + 1; 

      if(evlGridSize >= EVLGRIDSIZEmax){
          std::cout << "the evl size ("<<evlGridSize
                    <<") is larger than the maximum size allowed. Increase metaparameter's grid_SizeMax ("<< EVLGRIDSIZEmax <<") to allow larger evl tissue." << std::endl;
          sd->errorCode[0] = 103;
      }

      // // empty grid box cells counter
      thrust::fill_n( 
                (*sd).customState.evlGridPartNum.begin(),
                evlGridSize * evlGridSize * evlGridSize, 
                0);

      thrust::for_each(
          first_evlPart,
          last_evlPart,
          fill_grid(
              thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),  
              evlGridBoxSize,
              evlworldMin,
              evlGridSize,
              thrust::raw_pointer_cast(&((*sd).customState.evlGridPartNum[0])),
              thrust::raw_pointer_cast(&((*sd).customState.evlGridPartId[0])),
              EVLGRIDNUMPARTmax,
              thrust::raw_pointer_cast(&((*sd).errorCode[0])),
              currentTimeStep
          )
      );

      thrust::for_each(
          first_evlPart,
          last_evlPart,
          custom_evl_metric_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            evlGridBoxSize,
            evlworldMin,
            evlGridSize,
            thrust::raw_pointer_cast(&((*sd).customState.evlGridPartNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlGridPartId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      thrust::for_each(
          first_evlPart,
          last_evlPart,
          custom_evl_topological_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /*********  Cells-EVL neighborhood *********/
      /*******************************************/
      /*******************************************/

      thrust::for_each(
          first_cell,
          last_cell,
          custom_cellsevl_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlNormal[0])),
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            p->customParam[0].evlRadiusAB,
            evlGridBoxSize,
            evlworldMin,
            evlGridSize,
            thrust::raw_pointer_cast(&((*sd).customState.evlGridPartNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlGridPartId[0])),
            thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.cellsEvlNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.cellsEvlNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /******* Yolk margin-EVL neighborhood ******/
      /*******************************************/
      /*******************************************/

      thrust::for_each(
          first_yolkPart,
          first_yolkInteriorPart,
          custom_yolkmarginevl_metric_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlRadius[0])),
            evlGridBoxSize,
            evlworldMin,
            evlGridSize,
            thrust::raw_pointer_cast(&((*sd).customState.evlGridPartNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlGridPartId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneEYSL[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      thrust::for_each(
          first_yolkPart,
          first_yolkInteriorPart,
          custom_yolkmarginevl_topological_neighborhood(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /************  EVL Polarization ************/
      /*******************************************/
      /*******************************************/

      thrust::for_each(
          first_evlPart,
          last_evlPart,
          updateEVLNormales(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlNormal[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );
    }

    void custom_algo_forces(
              MetaParam<HOST>* mp, 
              Param_Host * p, 
              Param_Device * pd, 
              State_Host * s, 
              State_Device * sd, 
              int currentTimeStep,
              uint loop)
    {

      thrust::counting_iterator<int> first_yolkPart(0);     
      thrust::counting_iterator<int> first_yolkInteriorPart = first_yolkPart + NUMPARTYOLKMEMBRANE;     
      thrust::counting_iterator<int> last_yolkPart = first_yolkPart + NUMPARTYOLK;

      uint numEVLCells = sd->customState.numPartEVL[0];
      
      thrust::counting_iterator<int> first_evlPart(0);     
      thrust::counting_iterator<int> last_evlPart = first_evlPart + numEVLCells;

      thrust::counting_iterator<int> first_cell(0);       
      thrust::counting_iterator<int> last_cell = first_cell + s->numCells[0];

      thrust::fill_n((*sd).customState.yolkForcesNum.begin(), NUMPARTYOLK, 0);
      thrust::fill_n((*sd).customState.evlForcesNum.begin(), numEVLCells, 0);

      /**********************/
      /**********************/
      /*** YOLK Forces ******/
      /**********************/
      /**********************/

      thrust::for_each(
          first_yolkPart,
          last_yolkPart,
          custom_yolk_forces_computation(
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            s->customState.yolkInteriorRadius[0],
            thrust::raw_pointer_cast(&((*pd).customParam[0])),
            s->customState.yolkInteriorDistEq[0],
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkInteriorTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneNeighbRL[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /*********  Yolk-Cells Forces       ********/
      /*******************************************/
      /*******************************************/

      thrust::fill_n((*sd).customState.yolkMembraneEYSLupdate.begin(), NUMPARTYOLKMEMBRANE, 0);

      thrust::for_each(
          first_yolkPart,
          first_yolkInteriorPart,
          custom_yolkcells_forces_computation(
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneActivated[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkCellsNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkCellsNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            s->customState.yolkInteriorRadius[0],
            p->customParam[0].cellsYolkCmax,
            p->customParam[0].cellsYolkSurfaceScaling,
            p->customParam[0].cellsYolkEquilibriumDistance,
            p->customParam[0].cellsYolkAttractionCoefficient,
            p->customParam[0].cellsYolkRepulsionCoefficient,
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneEYSL[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneTangentParams[0])),
            p->customParam[0].marginResistance,
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneNextNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneNextId[0])),
            thrust::raw_pointer_cast(&((*sd).cellForces[0])),
            thrust::raw_pointer_cast(&((*sd).cellForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneEYSLupdate[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /**********************/
      /**********************/
      /*******  EVL  ********/
      /**********************/
      /**********************/

      thrust::for_each(
          first_evlPart,
          last_evlPart,
          custom_evl_forces_computation(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlRadius[0])),
            p->customParam[0].evlRLCoeff,
            p->customParam[0].evlStiffness,
            thrust::raw_pointer_cast(&((*sd).customState.evlTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlPressure[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /*********  Cells-EVL Forces       *********/
      /*******************************************/
      /*******************************************/

      thrust::for_each(
          first_cell,
          last_cell,
          custom_cellsevl_forces_computation(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlNormal[0])),
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.cellsEvlNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.cellsEvlNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            p->customParam[0].evlRadiusAB,
            p->customParam[0].cellsEvlCmax,
            p->customParam[0].cellsEvlSurfaceScaling,
            p->customParam[0].cellsEvlEquilibriumDistance,
            p->customParam[0].cellsEvlAttractionCoefficient,
            p->customParam[0].cellsEvlRepulsionCoefficient,
            thrust::raw_pointer_cast(&((*sd).cellForces[0])),
            thrust::raw_pointer_cast(&((*sd).cellForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /******* Yolk margin-EVL forces ************/
      /*******************************************/
      /*******************************************/

      thrust::for_each(
          first_yolkPart,
          first_yolkInteriorPart,
          custom_yolkmarginevl_forces_computation(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMarginEvlTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlRadius[0])),
            s->customState.yolkInteriorRadius[0],
            p->customParam[0].yolkMarginEvlStiffness,
            thrust::raw_pointer_cast(&((*sd).customState.yolkForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );      

      /*******************************************/
      /*******************************************/
      /*************  Yolk Integration  **********/
      /*******************************************/
      /*******************************************/

      thrust::for_each(
          first_yolkPart,
          last_yolkPart,
          custom_yolk_forces_integration(
            thrust::raw_pointer_cast(&((*sd).customState.yolkPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkForcesNum[0])),
            s->customState.yolkInteriorRadius[0],
            s->customState.yolkMembraneRadius[0],
            p->globalDamping[0],
            loop,
            thrust::raw_pointer_cast(&((*sd).customState.yolk_Runge_Kutta_K[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolk_Runge_Kutta_InitPos[0])),
            p->deltaTime[0],
            mp->spatialBorderMin[0],
            mp->spatialBorderMax[0],
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneEYSL[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneEYSLupdate[0])),
            thrust::raw_pointer_cast(&((*sd).customState.yolkMembraneActivated[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
          )
      );

      /*******************************************/
      /*******************************************/
      /*************  EVL Integration  ***********/
      /*******************************************/
      /*******************************************/

      thrust::for_each(
          first_evlPart,
          last_evlPart,
          custom_evl_forces_integration(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForces[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlForcesNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlRadius[0])),
            p->globalDamping[0],
            loop,
            thrust::raw_pointer_cast(&((*sd).customState.evl_Runge_Kutta_K[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evl_Runge_Kutta_InitPos[0])),
            p->deltaTime[0],
            mp->spatialBorderMin[0],
            mp->spatialBorderMax[0],
            p->customParam[0].evlRadiusAB
          )
      );
    }

    void custom_algo_evl_growth_division(
              MetaParam<HOST>* mp, 
              Param_Host * p, 
              Param_Device * pd, 
              State_Host * s, 
              State_Device * sd, 
              int currentTimeStep)
    {

      uint numEVLCells = sd->customState.numPartEVL[0];
      
      thrust::counting_iterator<int> first_evlPart(0);     
      thrust::counting_iterator<int> last_evlPart = first_evlPart + numEVLCells;

      thrust::for_each(
          first_evlPart,
          last_evlPart,
          custom_evl_growth_division(
            thrust::raw_pointer_cast(&((*sd).customState.evlPosition[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlNormal[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlRadius[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlTimer[0])),
            thrust::raw_pointer_cast(&((*sd).customState.evlPressure[0])),
            p->cellCycleParams[0].mPhaseLength,
            p->customParam[0].evlGrowthThreshold,
            p->customParam[0].evlLateralGrowthRatio,
            p->customParam[0].evlRadiusLimit,
            thrust::raw_pointer_cast(&((*pd).randomUniform[0])),
            thrust::raw_pointer_cast(&((*sd).randomUniform_Counter[0])),
            currentTimeStep,
            thrust::raw_pointer_cast(&((*sd).customState.numPartEVL[0]))
          )
      );
      
    }

    void custom_algo_yolk_evl_diffusion(
              MetaParam<HOST>* mp, 
              Param_Host * p, 
              Param_Device * pd, 
              State_Host * s, 
              State_Device * sd, 
              int currentTimeStep)
    {
      // EVL and Yolk acts as source and sink
      thrust::counting_iterator<int> first_cell(0);       
      thrust::counting_iterator<int> last_cell = first_cell + s->numCells[0];

      thrust::for_each(
          first_cell,
          last_cell,
          custom_yolk_evl_diffusion(
            thrust::raw_pointer_cast(&((*sd).customState.cellsYolkNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).customState.cellsEvlNeighbNum[0])),
            p->customParam[0].yolkLigandId,
            p->customParam[0].evlLigandId,
            p->customParam[0].yolkLigandUpdate,
            p->customParam[0].evlLigandUpdate,
            thrust::raw_pointer_cast(&((*sd).cellLigand[0])),
            thrust::raw_pointer_cast(&((*sd).cellLigandUpdate[0]))
          )
      );

    }

}
