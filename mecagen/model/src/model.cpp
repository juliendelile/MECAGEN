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

// Class defnition:
#include "model.hpp"
    
// Dependencies:
#include "param_host.hpp"
#include "param_device.hpp"
#include "state_host.hpp"
#include "state_device.hpp"

#include "cellshape.hpp"
#include "spatialNeighborhood.hpp"
#include "cellbehavior.hpp"
#include "diffusion.hpp"

#include "grn.hpp"

#include "polarization.hpp"
#include "forces.hpp"
#include "mitosis.hpp"

// Standard:
#include <iostream> //cout

#define __THRUST_SYNCHRONOUS

namespace mg {

  int Model::algoStep1(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep){


    // cudaProfilerStart();

    //Get required host value from device
    s->numCells[0] = sd->numCells[0];

    s->currentTimeStep[0] = sd->currentTimeStep[0] = currentTimeStep;

    std::cout << "Timestep : " << currentTimeStep 
                << "    Numcells : " << s->numCells[0]
                << std::endl;

    thrust::counting_iterator<int> first_cell(0);       
    thrust::counting_iterator<int> last_cell = first_cell + s->numCells[0];

    //Reset random number courters if needed
    if(sd->randomGaussian_Counter[0] + s->numCells[0] >= 400000 ){
      sd->randomGaussian_Counter[0] = 0;
    }
    if(sd->randomUniform_Counter[0] + s->numCells[0] >= 400000 ){
      sd->randomUniform_Counter[0] = 0;
    }
    
    /******* Data center *****/

    // The data center value need to be calculated for the spatial ligand source specification
    if(currentTimeStep%100==0){
      d3 center = thrust::reduce(
            sd->cellPosition.begin(), 
            sd->cellPosition.begin() + s->numCells[0],
            (d3) d3(.0),
            thrust::plus<d3>()
            );
      sd->embryoCenter[0] = s->embryoCenter[0] = center / (double)s->numCells[0];
    }

    s->max_c_max = *std::max_element(
                      p->mechaParams[0].maximumDistanceCoefficient,
                      p->mechaParams[0].maximumDistanceCoefficient+9
                      );

    //determine celltype (0 undef, 1 mes, 2 epi) from GRN state
    thrust::for_each(
        first_cell,
        last_cell,
        attributeCellTypeFromGRNstate(
            thrust::raw_pointer_cast(pd->cellTypeNodes.data()),
            thrust::raw_pointer_cast(&((*sd).cellProtein[0])),
            thrust::raw_pointer_cast(&((*sd).cellType[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0])),
            thrust::raw_pointer_cast(&((*sd).cellEpiIsPolarized[0])),
            thrust::raw_pointer_cast(&((*sd).cellEpiId[0]))
        )
    );

    #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
      cudaDeviceSynchronize();
    #endif

    // *********************************
    // *********************************
    // *******   Cell Shape   **********
    // *********************************
    // *********************************

    // Compute cell total surface 
    // If a cell is epithelial with apicobasal polarization, we compute the border separating 
    // the lateral and apicobasal domains (shaperatio)
    thrust::for_each(
      first_cell,
      last_cell,
      evaluateCellShape(
        thrust::raw_pointer_cast(&((*sd).cellType[0])),
        thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
        thrust::raw_pointer_cast(&((*sd).cellEpiIsPolarized[0])),
        thrust::raw_pointer_cast(&((*pd).randomUniform[0])),
        thrust::raw_pointer_cast(&((*sd).randomUniform_Counter[0])),
        thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
        thrust::raw_pointer_cast(&((*sd).cellSurface[0])),
        thrust::raw_pointer_cast(&((*sd).cellShapeRatio[0])),
        thrust::raw_pointer_cast(&((*sd).errorCode[0]))
      )
    );

    #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
      cudaDeviceSynchronize();
    #endif
    
    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }

    // *********************************
    // *********************************
    // *****    Neighborhood     *******
    // *********************************
    // *********************************

    // *** Calibrate world grid ********

    // get Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, CellRadiusMax
    boundingBox init = boundingBox(sd->cellPosition[0],sd->cellPosition[0]);

    bbox_reduction binary_op;

    boundingBox bbox = thrust::reduce(
                    (*sd).cellPosition.begin(),
                    (*sd).cellPosition.begin() + s->numCells[0],
                    init,
                    binary_op);

    thrust::device_vector<d3>::iterator radMax = thrust::max_element(
                            (*sd).cellRadius.begin(),
                            (*sd).cellRadius.begin() + s->numCells[0],
                            compare_first_radius_value());

    d3 radmax = *radMax;

    sd->gridBoxSize[0] = s->gridBoxSize[0] = 2.0 * s->max_c_max * radmax.x;
    
    sd->worldMax[0] = s->worldMax[0] = std::max(std::max(bbox.upper_right.x, bbox.upper_right.y), bbox.upper_right.z);

    sd->worldOrigin[0] = s->worldOrigin[0] = std::min(std::min(bbox.lower_left.x, bbox.lower_left.y), bbox.lower_left.z);

    sd->worldSize[0] = s->worldSize[0] = s->worldMax[0] - s->worldOrigin[0];
    
    sd->gridSize[0] = s->gridSize[0] = std::floor(s->worldSize[0] / s->gridBoxSize[0]) + 1;
    
    // check that the number of boxes in the grid is less than the maximum number allowed
    if(s->gridSize[0] >= mp->grid_SizeMax[0]){
        std::cout << "the cell swarm length in larger than the maximum size allowed. Increase metaparameter's grid_SizeMax to allow larger swarms." << std::endl;
        sd->errorCode[0] = 1;
        return sd->errorCode[0];
    }

    // empty grid box cells counter
    thrust::fill_n((*sd).gridPartNum.begin(), s->gridSize[0] * s->gridSize[0] * s->gridSize[0], 0);

    // fill grid box cells counters
    #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
      cudaDeviceSynchronize();
    #endif

    thrust::for_each(
        first_cell,
        last_cell,
        fill_grid(
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            s->gridBoxSize[0],
            s->worldOrigin[0],
            s->gridSize[0],
            thrust::raw_pointer_cast(&((*sd).gridPartNum[0])),
            thrust::raw_pointer_cast(&((*sd).gridPartId[0])),
            mp->gridBox_NumPartMax[0],
            thrust::raw_pointer_cast(&((*sd).errorCode[0])),
            currentTimeStep
        )
    );

    if(sd->errorCode[0] != 0){
        return sd->errorCode[0];
    }

    #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaDeviceSynchronize();
    #endif

    // ***** Metric neighb. ************
    // select metric neighbors for each particles
    thrust::for_each(
        first_cell,
        last_cell,
        metric_neighborhood(
            s->numCells[0],
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellType[0])),
            thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            s->gridBoxSize[0],
            s->worldOrigin[0],
            s->gridSize[0],
            thrust::raw_pointer_cast(&((*sd).gridPartNum[0])),
            thrust::raw_pointer_cast(&((*sd).gridPartId[0])),
            mp->gridBox_NumPartMax[0],
            mp->numNeighbMax[0],
            currentTimeStep,
            thrust::raw_pointer_cast(&((*sd).cellShapeRatio[0])),
            thrust::raw_pointer_cast(&((*sd).cellMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).cellMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).cellMetricNeighbAngle[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0])),
            thrust::raw_pointer_cast(&((*pd).mechaParams[0]))
        )
    );

    if(sd->errorCode[0] != 0){
        return sd->errorCode[0];
    }

    // ***** Topological neighb. *******
    // trim the metric neighborhood according to topological criteria
    #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaDeviceSynchronize();
    #endif

    thrust::for_each(
        first_cell,
        last_cell,
        new_topological_neighborhood(
            thrust::raw_pointer_cast(&((*sd).cellType[0])),
            thrust::raw_pointer_cast(&((*sd).cellMetricNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).cellMetricNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),
            thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).cellMetricNeighbAngle[0])),
            mp->numNeighbMax[0],
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
            currentTimeStep,
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
        )
    );

    if(sd->errorCode[0] != 0){
        return sd->errorCode[0];
    }

    CUSTOM_ALGO_NEIGHB

    // cudaProfilerStop();
    
    // *********************************
    // *********************************
    // *********  Polarization I *******
    // *********************************
    // *********************************
    thrust::for_each(
            first_cell,
            last_cell,
            computePolarizationAxes(
                      p->numPolarizationAxes[0],
                      p->numLigands[0],
                      thrust::raw_pointer_cast(&((*pd).polarizationAxisParams[0])),  
                      thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
                      thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),   
                      thrust::raw_pointer_cast(&((*sd).cellLigand[0])),
                      thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
                      thrust::raw_pointer_cast(&((*sd).cellCandidateAxes[0])),
                      thrust::raw_pointer_cast(&((*sd).cellType[0])),
                      thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
                      thrust::raw_pointer_cast(&((*sd).cellEpiIsPolarized[0])),
                      thrust::raw_pointer_cast(&((*sd).cellCandidateAxesUpdate[0]))
                )
    );

    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }

    thrust::for_each(
            first_cell,
            last_cell,
            updatePolarizationAxes(
                      p->numPolarizationAxes[0],
                      thrust::raw_pointer_cast(&((*sd).cellCandidateAxes[0])),
                      thrust::raw_pointer_cast(&((*sd).cellCandidateAxesUpdate[0]))
                )
    );

    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }

    // *********************************
    // *********************************
    // *****  Meca State Setting  ******
    // *********************************
    // *********************************

    // ***** Set meca state. *******
    thrust::for_each(
        first_cell,
        last_cell,
        setMecaState_GRN(
            thrust::raw_pointer_cast(&((*sd).cellType[0])),
            thrust::raw_pointer_cast(&((*sd).cellProtein[0])),
            thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
            thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),   
            thrust::raw_pointer_cast(&((*sd).cellEpiIsPolarized[0])),   
            thrust::raw_pointer_cast(&((*sd).cellNeighbIsLateral[0])),   
            thrust::raw_pointer_cast(&((*sd).cellCandidateAxes[0])),
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            thrust::raw_pointer_cast(&((*sd).cellAxis1[0])),
            thrust::raw_pointer_cast(&((*sd).cellShapeRatio[0])),
            thrust::raw_pointer_cast(&((*sd).cellSurface[0])),
            thrust::raw_pointer_cast(&((*sd).cellProtrusionExtForce[0])),
            p->numPolarizationNodes[0],
            thrust::raw_pointer_cast(&((*pd).polarizationNodes[0])),
            thrust::raw_pointer_cast(&((*pd).forcePolarizationNode[0])),
            thrust::raw_pointer_cast(&((*pd).protrusionNode[0])),
            thrust::raw_pointer_cast(&((*pd).bipolarityNode[0])),
            p->numAdhesionNodes[0],
            thrust::raw_pointer_cast(&((*pd).adhesionNodes[0])),
            p->numEpiPolarizationNodes[0],
            thrust::raw_pointer_cast(&((*pd).epiPolarizationNodes[0])),
            thrust::raw_pointer_cast(&((*pd).mechaParams[0])),
            thrust::raw_pointer_cast(&((*pd).randomUniform[0])),
            thrust::raw_pointer_cast(&((*sd).randomUniform_Counter[0])),
            thrust::raw_pointer_cast(&((*sd).cellContactSurfaceArea[0])),
            thrust::raw_pointer_cast(&((*sd).cellEquilibriumDistance[0])),
            thrust::raw_pointer_cast(&((*sd).cellAttractionCoefficient[0])),
            thrust::raw_pointer_cast(&((*sd).cellRepulsionCoefficient[0])),
            thrust::raw_pointer_cast(&((*sd).cellPlanarRigidityCoefficient[0])),
            thrust::raw_pointer_cast(&((*sd).cellIntercalateWithNeighb[0])),
            thrust::raw_pointer_cast(&((*sd).cellIntercalationIntensity[0])),
            thrust::raw_pointer_cast(&((*sd).cellIntercalationBipolar[0])),
            thrust::raw_pointer_cast(&((*sd).cellApicalConstrictionWithNeighb[0])),
            thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
            thrust::raw_pointer_cast(&((*sd).cellBlebbingMode[0])),
            thrust::raw_pointer_cast(&((*sd).cellRandomBlebbingAxis[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0]))
        )
    );

    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }

    // *********************************
    // *********************************
    // *****      Diffusion      *******
    // *********************************
    // *********************************
  
    //requires cellContactSurfaceArea from setMeca function
    thrust::for_each(
            first_cell,
            last_cell,
            diffuseLigands(
                    p->numLigands[0],    
                    thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
                    thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),   
                    thrust::raw_pointer_cast(&((*pd).ligandParams[0])),   
                    thrust::raw_pointer_cast(&((*sd).cellLigand[0])),
                    thrust::raw_pointer_cast(&((*sd).cellLigandUpdate[0])),
                    thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
                    thrust::raw_pointer_cast(&((*sd).cellContactSurfaceArea[0])),
                    p->deltaTime[0]
                )
    );

    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }

    thrust::for_each(
            first_cell,
            last_cell,
            computeGRN(   
                    thrust::raw_pointer_cast(&((*sd).cellProteinUpdate[0])),
                    thrust::raw_pointer_cast(&((*sd).cellLigandUpdate[0])),
                    thrust::raw_pointer_cast(&((*sd).errorCode[0])),
                    thrust::raw_pointer_cast(&((*sd).cellProtein[0])),
                    thrust::raw_pointer_cast(&((*sd).cellLigand[0])),
                    p->numProteins[0],
                    thrust::raw_pointer_cast(&((*pd).proteins[0])),
                    p->numGenes[0],
                    thrust::raw_pointer_cast(&((*pd).genes[0])),
                    p->numReceptors[0],
                    thrust::raw_pointer_cast(&((*pd).receptors[0])),
                    p->numSecretors[0],
                    thrust::raw_pointer_cast(&((*pd).secretors[0])),
                    p->numPPInteractions[0],
                    thrust::raw_pointer_cast(&((*pd).ppInteractions[0])),
                    p->numProteinNodes[0],
                    thrust::raw_pointer_cast(&((*pd).proteinNodes[0])),
                    thrust::raw_pointer_cast(&((*pd).mechanoSensorNode[0])),
                    thrust::raw_pointer_cast(&((*sd).cellMechanoSensorQ[0])),
                    thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
                    currentTimeStep,
                    p->deltaTime[0],
                    thrust::raw_pointer_cast(&((*sd).embryoCenter[0]))
                )
    );

    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }


    thrust::for_each(
            first_cell,
            last_cell,
            transmembraneLigandSensing(
              p->numTransReceptors[0],
              thrust::raw_pointer_cast(&((*pd).transReceptors[0])),
              thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
              thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),
              thrust::raw_pointer_cast(&((*sd).cellSurface[0])),
              thrust::raw_pointer_cast(&((*sd).cellContactSurfaceArea[0])),
              thrust::raw_pointer_cast(&((*sd).cellProtein[0])),
              thrust::raw_pointer_cast(&((*sd).cellProteinUpdate[0])),
              thrust::raw_pointer_cast(&((*sd).cellLigand[0])),
              p->deltaTime[0],
              thrust::raw_pointer_cast(&((*sd).errorCode[0]))
            )
    );


    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }

    CUSTOM_ALGO_REGULATION

    thrust::for_each(
            first_cell,
            last_cell,
            updateProteins(
                    p->numProteins[0],    
                    thrust::raw_pointer_cast(&((*pd).proteins[0])),   
                    thrust::raw_pointer_cast(&((*sd).cellProtein[0])),
                    thrust::raw_pointer_cast(&((*sd).cellProteinUpdate[0])),
                    thrust::raw_pointer_cast(&((*sd).errorCode[0]))
                )
    );

    // Cell ligand quantities are updated in grn function (signal transduction and secretion)
    thrust::for_each(
            first_cell,
            last_cell,
            updateLigands(
                    p->numLigands[0],    
                    thrust::raw_pointer_cast(&((*pd).ligandParams[0])),   
                    thrust::raw_pointer_cast(&((*sd).cellLigand[0])),
                    thrust::raw_pointer_cast(&((*sd).cellLigandUpdate[0]))
                )
    );

    return sd->errorCode[0];
}

int Model::algoStep2(MetaParam<HOST>* mp, Param_Host * p, Param_Device * pd, State_Host * s, State_Device * sd, int currentTimeStep){
    
    // cudaProfilerStart();

    thrust::counting_iterator<int> first_cell(0);
    thrust::counting_iterator<int> last_cell = first_cell + s->numCells[0];

    // // *********************************
    // // *********************************
    // // *****    Biomechanics     *******
    // // *********************************
    // // *********************************

    // // Runge Kutta integration, we iterate through the following methods (forces_computation and forces_integration) 4 times.
    uint loop = 0;

    while(loop < 4){
     
        // ****** Forces computation *******
        
        // empty force counter
        thrust::fill_n((*sd).cellForcesNum.begin(), s->numCells[0], 0);
        thrust::fill_n((*sd).cellProtrusionExtForcesNum.begin(), s->numCells[0], 0);
        thrust::fill_n((*sd).cellMechanoSensorQsNum.begin(), s->numCells[0], 0);

        // computes forces from the topological neighbor list
        #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
            cudaDeviceSynchronize();
        #endif
        
        thrust::for_each(
            first_cell,
            last_cell,
            forces_computation(
                thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
                thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
                thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),
                thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
                thrust::raw_pointer_cast(&((*sd).cellNeighbIsLateral[0])),
                thrust::raw_pointer_cast(&((*sd).cellEpiIsPolarized[0])),
                thrust::raw_pointer_cast(&((*sd).cellEpiId[0])),
                thrust::raw_pointer_cast(&((*sd).cellAxis1[0])),
                thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
                thrust::raw_pointer_cast(&((*sd).cellType[0])),
                thrust::raw_pointer_cast(&((*sd).cellContactSurfaceArea[0])),
                thrust::raw_pointer_cast(&((*sd).cellEquilibriumDistance[0])),
                thrust::raw_pointer_cast(&((*sd).cellAttractionCoefficient[0])),
                thrust::raw_pointer_cast(&((*sd).cellRepulsionCoefficient[0])),
                thrust::raw_pointer_cast(&((*sd).cellPlanarRigidityCoefficient[0])),
                thrust::raw_pointer_cast(&((*sd).cellIntercalateWithNeighb[0])),
                thrust::raw_pointer_cast(&((*sd).cellIntercalationIntensity[0])),
                thrust::raw_pointer_cast(&((*sd).cellIntercalationBipolar[0])),
                thrust::raw_pointer_cast(&((*sd).cellApicalConstrictionWithNeighb[0])),
                thrust::raw_pointer_cast(&((*sd).cellForces[0])),
                thrust::raw_pointer_cast(&((*sd).cellForcesNum[0])),
                thrust::raw_pointer_cast(&((*sd).cellProtrusionExtForces[0])),
                thrust::raw_pointer_cast(&((*sd).cellProtrusionExtForcesNum[0])),
                thrust::raw_pointer_cast(&((*sd).cellMechanoSensorQs[0])),
                thrust::raw_pointer_cast(&((*sd).cellMechanoSensorQsNum[0])),
                thrust::raw_pointer_cast(&((*sd).errorCode[0]))
            )
        );
    

        if(sd->errorCode[0] != 0){
            return sd->errorCode[0];
        }

        CUSTOM_ALGO_FORCES

        // ****** Forces integration *******
        #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
            cudaDeviceSynchronize();
        #endif
        
        thrust::for_each(
            first_cell,
            last_cell,
            forces_integration(
                thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
                thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
                thrust::raw_pointer_cast(&((*sd).cellForces[0])),
                thrust::raw_pointer_cast(&((*sd).cellForcesNum[0])),
                thrust::raw_pointer_cast(&((*sd).cellProtrusionExtForce[0])),
                thrust::raw_pointer_cast(&((*sd).cellProtrusionExtForces[0])),
                thrust::raw_pointer_cast(&((*sd).cellProtrusionExtForcesNum[0])),
                p->globalDamping[0],
                loop,
                thrust::raw_pointer_cast(&((*sd).runge_Kutta_K[0])),
                thrust::raw_pointer_cast(&((*sd).runge_Kutta_K_Protr_Ext[0])),
                thrust::raw_pointer_cast(&((*sd).runge_Kutta_K_Mecha_Sensor_Q[0])),
                thrust::raw_pointer_cast(&((*sd).runge_Kutta_InitPos[0])),
                thrust::raw_pointer_cast(&((*sd).cellMechanoSensorQ[0])),
                thrust::raw_pointer_cast(&((*sd).cellMechanoSensorQs[0])),
                thrust::raw_pointer_cast(&((*sd).cellMechanoSensorQsNum[0])),
                mp->numCellsMax[0],
                p->deltaTime[0],
                thrust::raw_pointer_cast(&((*sd).cellType[0])),
                thrust::raw_pointer_cast(&((*sd).cellSurface[0])),
                mp->spatialBorderMin[0],
                mp->spatialBorderMax[0]
            )
        );

        if(sd->errorCode[0] != 0){
            return sd->errorCode[0];
        }

        loop++;
    }


    // *********************************
    // *********************************
    // ********  Polarization II *******
    // *********************************
    // *********************************

    // The apicobasal polarization axis is not updated before force computation as the AB axis is used 
    // in the neighborhood module and forces should not computed with a different axis.
    thrust::for_each(
      first_cell,
      last_cell,
      evaluateEpithelialApicoBasalPolarity(
        thrust::raw_pointer_cast(&((*sd).cellType[0])),
        thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbNum[0])),
        thrust::raw_pointer_cast(&((*sd).cellTopologicalNeighbId[0])),
        thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
        thrust::raw_pointer_cast(&((*sd).cellNeighbIsLateral[0])),
        thrust::raw_pointer_cast(&((*sd).cellEpiIsPolarized[0])),
        thrust::raw_pointer_cast(&((*sd).cellEpiId[0])),
        thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
        thrust::raw_pointer_cast(&((*sd).errorCode[0])),
        currentTimeStep
      )
    );

    if(sd->errorCode[0] != 0){
      return sd->errorCode[0];
    }


    // // *********************************
    // // *********************************
    // // *****    Cell Cycle       *******
    // // *********************************
    // // ********************************* 

    thrust::for_each(
        first_cell,
        last_cell,
        manage_mitosis(
            thrust::raw_pointer_cast(&((*sd).cellPosition[0])),
            thrust::raw_pointer_cast(&((*sd).cellAxisAB[0])),
            thrust::raw_pointer_cast(&((*sd).cellRadius[0])),
            thrust::raw_pointer_cast(&((*sd).cellState[0])),
            thrust::raw_pointer_cast(&((*sd).cellTimer[0])),
            thrust::raw_pointer_cast(&((*sd).cellGeneration[0])),
            thrust::raw_pointer_cast(&((*sd).cellCycleLength[0])),
            currentTimeStep,
            p->cellCycleParams[0].mPhaseLength,
            thrust::raw_pointer_cast(&((*sd).numCells[0])),
            thrust::raw_pointer_cast(&((*pd).randomGaussian[0])),
            thrust::raw_pointer_cast(&((*sd).randomGaussian_Counter[0])),
            thrust::raw_pointer_cast(&((*pd).randomUniform[0])),
            thrust::raw_pointer_cast(&((*sd).randomUniform_Counter[0])),
            p->deltaTime[0],
            thrust::raw_pointer_cast(&((*sd).cellId_bits[0])),
            thrust::raw_pointer_cast(&((*sd).errorCode[0])),
            thrust::raw_pointer_cast(&((*pd).customParam[0])),
            thrust::raw_pointer_cast(&((*pd).cellCycleParams[0])),
            p->numLigands[0],
            thrust::raw_pointer_cast(&((*sd).cellLigand[0])),
            thrust::raw_pointer_cast(&((*sd).cellType[0])),
            thrust::raw_pointer_cast(&((*sd).cellEpiIsPolarized[0])),
            thrust::raw_pointer_cast(&((*sd).cellProtrusionExtForce[0])),
            thrust::raw_pointer_cast(&((*sd).cellMechanoSensorQ[0])),
            p->numProteins[0],
            thrust::raw_pointer_cast(&((*sd).cellProtein[0]))
        )
    );

    CUSTOM_ALGO_MITOSIS

    // cudaProfilerStop();
    
    return sd->errorCode[0];
}

} // End namespace
