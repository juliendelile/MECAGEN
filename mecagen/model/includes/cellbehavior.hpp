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

#ifndef _CELLBEHAVIOR_H
#define _CELLBEHAVIOR_H

#include <cstring>
#include <stdio.h>	//printf
#include <assert.h>

namespace mg{

  /** Estimates the contact area between two cells of radii "radius1", "radius2", separated by a distance "dist" */
  inline
  __host__ __device__ double surface_estimation(
                            const double *dist,
                            const double *radius1,
                            const double *radius2,
                            const double *c_max,
                            const double *a
                            )
  { 
    double distmax = *c_max * (*radius1 + *radius2);
    return *a * (*dist - distmax) * (*dist - distmax);
  }

}

#include "grn.hpp"

namespace mg {


  /** This functor determines the cell archetype (mesenchymal, epithelial, or idle) according to the cell state.*/
	struct attributeCellTypeFromGRNstate
	{
    const RegulatoryElement*  cellTypeNodes;
    const double*             cellProtein;
    uint*					            cellType;
    uint*                     errorCode;
    uint*                     cellEpiIsPolarized;
    uint*                     cellEpiId;

    attributeCellTypeFromGRNstate(
                RegulatoryElement*    _cellTypeNodes,
                double*               _cellProtein,
                uint*					        _cellType,
                uint*                 _errorCode,
                uint*                 _cellEpiIsPolarized,
                uint*                 _cellEpiId
            )
         :  
            cellTypeNodes(_cellTypeNodes),
            cellProtein(_cellProtein),
            cellType(_cellType),
            errorCode(_errorCode),
            cellEpiIsPolarized(_cellEpiIsPolarized),
            cellEpiId(_cellEpiId)
            {}
		
		__device__
    void operator()(const int& idx){

      // Evaluate stateNode activity
      uint mesNode = isActive(&(cellTypeNodes[0]), &(cellProtein[idx * NUMPROTEINmax]));
      uint epiNode = isActive(&(cellTypeNodes[1]), &(cellProtein[idx * NUMPROTEINmax]));

      // Attribute cell type accordingly
      uint celltype;
      if (mesNode == 1 && epiNode == 1){
        printf("Error in GRN specification, cell %d can not be epithelial and mesenchymal simultaneously. Please check your GRN specification.\n",idx);
        errorCode[0] = 8;
        return;
      }
      else if(mesNode == 1){  //mesenchymal cell
        celltype = 1;
        //cell depolarization in case it was previously polarized
        cellEpiIsPolarized[idx] = 0;
      }
      else if(epiNode == 1){  //epithelial cell
        celltype = 2;     

        // Get epithelium ID
        cellEpiId[idx] = 0;
        for(int i=0; i<cellTypeNodes[1].numInputProtein; i++){
          uint idprot = cellTypeNodes[1].inputProteinID[i];
          if(
              cellProtein[idx*NUMPROTEINmax+idprot]
                > cellTypeNodes[1].inputThreshold[i]    
                ){ 
              cellEpiId[idx] = idprot;
          }
        }
      }
      else{                  // Undifferentiated cell
        celltype = 0;
        //cell depolarization in case it was previously polarized
        cellEpiIsPolarized[idx] = 0;
      }

    	cellType[idx] = celltype;
		}
	};

  /** This function calculates different adhesion mode (see "Zhang 2011 Computer Simulations of Cell Sorting Due to Differential Adhesion" "for details).*/
  inline
  __host__ __device__ double computeAdhesionCoefficient(
                  const AdhesionNode* adh, 
                  const double surf_conc_1, 
                  const double surf_conc_2
                  )
  {
    double conc;
    switch(adh->mode)
    {
      case 0: //Saturation model
        conc = adh->k_adh * fmin(surf_conc_1, surf_conc_2);
        break;
      case 1: //Trans-homophilic-bond model
        conc = adh->k_adh * surf_conc_1 * surf_conc_2;
        break;
      case 2: //Cis-dimer model
        conc = adh->k_adh 
                  * surf_conc_1 * surf_conc_1 
                  * surf_conc_2 * surf_conc_2;
        break;
      case 3: //"lazy" mode, adhesion is independent of the protein surface concentrations
        conc = adh->k_adh;
        break;
      default:
        printf("unknown adhesion function");
    }
    return conc;
  }

  /** This functor sets all the data required for the forces calculation from the cell state.*/
  struct setMecaState_GRN
  {
    const uint*             cellType;
    const double*           cellProtein;
    const uint*             cellTopologicalNeighbNum;
    const uint*             cellTopologicalNeighbId;
    uint*                   cellEpiIsPolarized;
    uint*                   cellNeighbIsLateral;
    const d3*               cellCandidateAxes;
    const d3*               cellPosition;
    const d3*               cellRadius;
    d3*                     cellAxis1;
    const double*           cellShapeRatio;
    const double*           cellSurface;
    const d3*               cellProtrusionExtForce;
    const uint              numPolarizationNodes;
    const PolarizationNode* polarizationNodes;
    const ForcePolarizationNode* forcePolarizationNode;
    const ProtrusionNode*   protrusionNode;
    const BipolarityNode*   bipolarityNode;
    const uint              numAdhesionNodes;
    const AdhesionNode*     adhesionNodes;
    const uint              numEpiPolarizationNodes;
    const uint*             epiPolarizationNodes;
    const MechaParams*      mechaParams;
    const double*           randomUniform;
    uint*                   randomUniform_Counter;
    double*                 cellContactSurfaceArea;
    double*                 cellEquilibriumDistance;
    double*                 cellAttractionCoefficient;
    double*                 cellRepulsionCoefficient;
    double*                 cellPlanarRigidityCoefficient;
    uint*                   cellIntercalateWithNeighb;
    double*                 cellIntercalationIntensity;
    uint*                   cellIntercalationBipolar;
    uint*                   cellApicalConstrictionWithNeighb;
    d3*                     cellAxisAB;
    uint*                   cellBlebbingMode;
    d3*                     cellRandomBlebbingAxis;
    uint*                   errorCode;
    
    setMecaState_GRN(
                uint*             _cellType,
                double*           _cellProtein,
                uint*             _cellTopologicalNeighbNum,
                uint*             _cellTopologicalNeighbId,
                uint*             _cellEpiIsPolarized,
                uint*             _cellNeighbIsLateral,
                d3*               _cellCandidateAxes,
                d3*               _cellPosition,
                d3*               _cellRadius,
                d3*               _cellAxis1,
                double*           _cellShapeRatio,
                double*           _cellSurface,
                d3*               _cellProtrusionExtForce,
                uint              _numPolarizationNodes,
                PolarizationNode* _polarizationNodes,
                ForcePolarizationNode* _forcePolarizationNode,
                ProtrusionNode*   _protrusionNode,
                BipolarityNode*   _bipolarityNode,
                uint              _numAdhesionNodes,
                AdhesionNode*     _adhesionNodes,
                uint              _numEpiPolarizationNodes,
                uint*             _epiPolarizationNodes,
                MechaParams*      _mechaParams,
                double*           _randomUniform,
                uint*             _randomUniform_Counter,
                double*           _cellContactSurfaceArea,
                double*           _cellEquilibriumDistance,
                double*           _cellAttractionCoefficient,
                double*           _cellRepulsionCoefficient,
                double*           _cellPlanarRigidityCoefficient,
                uint*             _cellIntercalateWithNeighb,
                double*           _cellIntercalationIntensity,
                uint*             _cellIntercalationBipolar,
                uint*             _cellApicalConstrictionWithNeighb,
                d3*               _cellAxisAB,
                uint*             _cellBlebbingMode,
                d3*               _cellRandomBlebbingAxis,
                uint*             _errorCode
            )
         :
            cellType(_cellType),
            cellProtein(_cellProtein),
            cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
            cellTopologicalNeighbId(_cellTopologicalNeighbId),
            cellEpiIsPolarized(_cellEpiIsPolarized),
            cellNeighbIsLateral(_cellNeighbIsLateral),
            cellCandidateAxes(_cellCandidateAxes),
            cellPosition(_cellPosition),
            cellRadius(_cellRadius),
            cellAxis1(_cellAxis1),
            cellShapeRatio(_cellShapeRatio),
            cellSurface(_cellSurface),
            cellProtrusionExtForce(_cellProtrusionExtForce),
            numPolarizationNodes(_numPolarizationNodes),
            polarizationNodes(_polarizationNodes),
            forcePolarizationNode(_forcePolarizationNode),
            protrusionNode(_protrusionNode),
            bipolarityNode(_bipolarityNode),
            numAdhesionNodes(_numAdhesionNodes),
            adhesionNodes(_adhesionNodes),
            numEpiPolarizationNodes(_numEpiPolarizationNodes),
            epiPolarizationNodes(_epiPolarizationNodes),
            mechaParams(_mechaParams),
            randomUniform(_randomUniform),
            randomUniform_Counter(_randomUniform_Counter),
            cellContactSurfaceArea(_cellContactSurfaceArea),
            cellEquilibriumDistance(_cellEquilibriumDistance),
            cellAttractionCoefficient(_cellAttractionCoefficient),
            cellRepulsionCoefficient(_cellRepulsionCoefficient),
            cellPlanarRigidityCoefficient(_cellPlanarRigidityCoefficient),
            cellIntercalateWithNeighb(_cellIntercalateWithNeighb),
            cellIntercalationIntensity(_cellIntercalationIntensity),
            cellIntercalationBipolar(_cellIntercalationBipolar),
            cellApicalConstrictionWithNeighb(_cellApicalConstrictionWithNeighb),
            cellAxisAB(_cellAxisAB),
            cellBlebbingMode(_cellBlebbingMode),
            cellRandomBlebbingAxis(_cellRandomBlebbingAxis),
            errorCode(_errorCode)
          {}

    // no __host__ here as we use the device method "myAtomic"
    __device__
    void operator()(const int& idx){

      uint celltype                 = cellType[idx];
      uint numNeighb                = cellTopologicalNeighbNum[idx];

      uint protrusion = 0;
      uint bipolar = 0;
      uint focalAdhesionID;
      double protrusionForce;

      // undifferentiated cell
      if(celltype == 0){
        // cell can have a behavioral axis only if it is differentiated
        cellBlebbingMode[idx] = 0;
        cellAxis1[idx] = d3(.0);
      }
      else{

        /************************************/
        /*** Polarization axis selection ****/
        /************************************/
        
        uint activePolarizationNodeCounter = 0;
        d3 activeAxis;

        // Protrusion-induced polarization axis
        d3 prot_ext_force = cellProtrusionExtForce[idx];
        if( isActive(&(forcePolarizationNode[0].regEl), &(cellProtein[idx*NUMPROTEINmax]))
            && length(prot_ext_force) > forcePolarizationNode[0].force_threshold ){
          activePolarizationNodeCounter++;
          activeAxis = prot_ext_force / length(prot_ext_force);
        }

        // gradient and cell-cell contact modes
        for(uint i=0; i < numPolarizationNodes; i++){
          if( isActive(&(polarizationNodes[i].regEl), &(cellProtein[idx*NUMPROTEINmax])) == 1 ){
            activePolarizationNodeCounter++;
            activeAxis = cellCandidateAxes[ idx * NUMAXESmax + polarizationNodes[i].axisID ];
          }
        }

        if(activePolarizationNodeCounter == 0){
          //Blebbing mode
          d3 currentAxis = cellAxis1[idx];
          d3 randomAxis = cellRandomBlebbingAxis[idx];
          // if cell was undifferentiated at the previous time step (only undifferentiated cell have null cellAxis1), 
          // or if the current axis is close enough from the target random axis, 
          // or if the randomAxis is not initialized (when simulation starts) !!! really ugly coding !!!,
          // -> we set up a new random axis.
          if( 
                length(currentAxis) == .0 
                || (fabs(dot(currentAxis, randomAxis)) < .1) 
                || length(randomAxis) != 1.0
            ){
            randomAxis = d3(
                            randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5,
                            randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5,
                            randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5
                          );
            randomAxis /= length(randomAxis);
            cellRandomBlebbingAxis[idx] = randomAxis;
          }
          currentAxis = currentAxis + .1 * randomAxis;
          cellAxis1[idx] = currentAxis / length(currentAxis);
          cellBlebbingMode[idx] = 1;
        }
        else if(activePolarizationNodeCounter == 1){
          //Axis selection via the active polarizationNode
          cellAxis1[idx] = activeAxis;
          cellBlebbingMode[idx] = 0;
        }

        /********************************/
        /**** Intercalation behavior ****/
        /********************************/
        
        if( isActive(&(protrusionNode[0].regEl), &(cellProtein[idx*NUMPROTEINmax])) ){
          if( activePolarizationNodeCounter == 1 ){
            protrusion = 1;
            if( isActive(&(bipolarityNode[0].regEl),&(cellProtein[idx*NUMPROTEINmax])) ){
              bipolar = 1;
            } 
            focalAdhesionID = protrusionNode[0].adhesionID;
            protrusionForce = protrusionNode[0].force;
          }
          else{
            printf("cell %d is trying to protrude, but no cell axis is activated. Please check your GRN specification.\n", idx);
          }
        }
        
        if(celltype == 2){

          if(cellEpiIsPolarized[idx] == 1){
            
            // /**************************************/
            // /**** Apical constriction behavior ****/
            // /**************************************/

            // uint numApicConstr  = stageParams->numApicalConstriction[waddCellType];
            // ApicalConstrictionParams acp;
            // if(numApicConstr != 0){
            //   // get apical params
            // }
          }
          else{

            /*****************************************************************/
            /*** AB Polarization axis candidate in unpolarized epithelium ****/
            /*****************************************************************/

            // Candidate axis, if any, is determine by the first active epithelial polarization node
            uint foundaxis = 0;
            d3 axis;
            for(uint i=0; i < numEpiPolarizationNodes; i++){
              
              uint polaNodeID = epiPolarizationNodes[i];

              if(isActive(&(polarizationNodes[polaNodeID].regEl), &(cellProtein[idx*NUMPROTEINmax])) == 1){

                uint axisID = polarizationNodes[polaNodeID].axisID;
                axis = cellCandidateAxes[ idx * NUMAXESmax + axisID ];

                if(length(axis) != .0){
                  foundaxis = 1;
                  break;
                }

              }
            }

            if(foundaxis){
              cellAxisAB[idx] = axis;
              cellEpiIsPolarized[idx] = 1;
            }

          }
        }
      }

      /**************************************/
      /**** State variable init ****/
      /**************************************/
      double wadh[NUMADHESIONNODEmax];

      d3 pos = cellPosition[idx], relPos, ABaxis;
      double radius1, radius1lat = cellRadius[idx].x, radius1ab, radius2, shapeRatio, dist;
      uint epi = (uint)(celltype == 2);

      double surface1 = cellSurface[idx];

      // If the cell is epithelial, the apicobasal radius is also taken into account
      if(epi){
        ABaxis = cellAxisAB[idx];
        shapeRatio = cellShapeRatio[idx];
        radius1ab = cellRadius[idx].y;
      }

      for(uint i=0; i<numNeighb; i++){
        
        uint topoNeighbIndex  = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId     = cellTopologicalNeighbId[topoNeighbIndex];
        uint celltype2        = cellType[neighbCellId];
        double surface2       = cellSurface[neighbCellId];

        relPos = cellPosition[neighbCellId] - pos;
        dist = length(relPos); 
        relPos /= dist;
        
        // Neighbor to neighbor surface and equilibrium distance depends on their size and deformation coefficients
        
        // Every cell have a single radius...
        if(!epi){
          radius1 = radius1lat;
        }
        // ... but epithelial cell with apicobasal polarization have two radii.
        // The selected radius depends whether the neighbor is lateral or not.
        else{
          double scal = dot( relPos, ABaxis );
          if( fabs(scal) < shapeRatio ){
            radius1 = radius1lat;
            cellNeighbIsLateral[topoNeighbIndex] = 1;
          }
          else{
            radius1 = radius1ab;
            cellNeighbIsLateral[topoNeighbIndex] = 0;
          }
        }
        

        // the same goes for the neighbor cell
        if(celltype2 != 2){
          radius2 = cellRadius[neighbCellId].x;
        }
        else{
          double scal = dot( relPos, cellAxisAB[neighbCellId] );
          if( fabs(scal) < cellShapeRatio[neighbCellId] ){
            radius2 = cellRadius[neighbCellId].x;
          }
          else{
            radius2 = cellRadius[neighbCellId].y;
          } 
        }

        double cmax         = mechaParams->maximumDistanceCoefficient[celltype * 3 + celltype2];
        double surfscaling  = mechaParams->surfaceScaling[celltype * 3 + celltype2];


        double Aij = surface_estimation(
                                    &dist,
                                    &radius1,
                                    &radius2,
                                    &cmax,
                                    &surfscaling
                                    );
        
        cellContactSurfaceArea[topoNeighbIndex] = Aij;
        
        double ceq = mechaParams->equilibriumDistanceCoefficient[celltype * 3 + celltype2];

        cellEquilibriumDistance[topoNeighbIndex] = ceq * (radius1+radius2);
        
        //Attraction coefficient
        double wadhsum = 0;
        
        for(uint j=0; j< numAdhesionNodes; j++){
          uint adhesionProt = adhesionNodes[j].proteinID;
          wadh[j] = computeAdhesionCoefficient(
                    &(adhesionNodes[j]), 
                    Aij / surface1 * cellProtein[idx*NUMPROTEINmax+adhesionProt], 
                    Aij / surface2 * cellProtein[neighbCellId*NUMPROTEINmax+adhesionProt]);
          
          wadhsum += wadh[j];
        }

        cellAttractionCoefficient[topoNeighbIndex]          = wadhsum;
        cellRepulsionCoefficient[topoNeighbIndex]           = mechaParams->repulsionCoefficient[celltype * 3 + celltype2];
        cellPlanarRigidityCoefficient[topoNeighbIndex]      = mechaParams->planarRigidityCoefficient[celltype * 3 + celltype2];

        if( protrusion == 1){
          cellIntercalateWithNeighb[topoNeighbIndex]        = 1;
          cellIntercalationIntensity[topoNeighbIndex]       = protrusionForce * wadh[focalAdhesionID];
          cellIntercalationBipolar[topoNeighbIndex]         = bipolar;
        }
      }

    } // end operator()
  }; // end functor 

} // end namespace    

#endif