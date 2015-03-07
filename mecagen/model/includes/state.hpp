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

#ifndef _STATE_
#define _STATE_

#include "metaparam.hpp"          
#include "thrust_objects.hpp"     // Backend<>
#include "custom.hpp"

namespace mg {

    // Forward declarations:
    template<int T> class MetaParam;

    /** State object.
    * This object must be deeply copiable, and follow the rule of three. Here, we use thrust vectors
    * as members so it is indeed the case. However, in the Host-Device mode of the framework, an additional method
    * must be implemented to allow the copy from the backend to the device backend ("copy" method).
    */
    template< int T >
    class State{

        public:

            /** State class constructor.
            * The thrust vectors are initialized in the constructor initializer list via the MetaParam object passed
            * as a parameter.
            */
            State(MetaParam<HOST>* mp):
                  metaParam(mp),
                  numCells(1),
                  currentTimeStep(1),
                  cellPosition(metaParam->numCellsMax[0]),
                  embryoCenter(1),
                  cellAxisAB(metaParam->numCellsMax[0]),
                  cellRadius(metaParam->numCellsMax[0]),
                  cellState(metaParam->numCellsMax[0]),
                  cellTimer(metaParam->numCellsMax[0]),
                  cellGeneration(metaParam->numCellsMax[0]),
                  cellCycleLength(metaParam->numCellsMax[0]),
                  cellId_bits(metaParam->numCellsMax[0]),
                  cellLigand(NUMLIGmax * metaParam->numCellsMax[0]),
                  cellEpiIsPolarized(metaParam->numCellsMax[0]),
                  cellProtrusionExtForce(metaParam->numCellsMax[0]),
                  cellMechanoSensorQ(metaParam->numCellsMax[0]),

                  randomGaussian_Counter(1),
                  randomUniform_Counter(1),
                  errorCode(1),

                  cellType(metaParam->numCellsMax[0]),
                  cellAxis1(metaParam->numCellsMax[0]),
                  gridBoxSize(1),
                  worldSize(1),
                  worldMax(1),
                  worldOrigin(1),
                  gridSize(1),
                  gridPartNum(metaParam->grid_SizeMax[0]*metaParam->grid_SizeMax[0]*metaParam->grid_SizeMax[0]),
                  gridPartId(metaParam->grid_SizeMax[0]*metaParam->grid_SizeMax[0]*metaParam->grid_SizeMax[0] * metaParam->gridBox_NumPartMax[0]),
                  cellMetricNeighbId(metaParam->numNeighbMax[0] * metaParam->numCellsMax[0]),
                  cellMetricNeighbNum(metaParam->numCellsMax[0]),
                  cellTopologicalNeighbId(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellTopologicalNeighbNum(metaParam->numCellsMax[0]),
                  cellMetricNeighbAngle(metaParam->numNeighbMax[0] * metaParam->numCellsMax[0]),
                  cellForces(NUMFORCEmax * metaParam->numCellsMax[0]),
                  cellProtrusionExtForces(NUMFORCEmax * metaParam->numCellsMax[0]),
                  cellMechanoSensorQs(NUMFORCEmax * metaParam->numCellsMax[0]),
                  cellForcesNum(metaParam->numCellsMax[0]),
                  cellProtrusionExtForcesNum(metaParam->numCellsMax[0]),
                  cellMechanoSensorQsNum(metaParam->numCellsMax[0]),
                  runge_Kutta_K(3 * metaParam->numCellsMax[0]),
                  runge_Kutta_K_Protr_Ext(3 * metaParam->numCellsMax[0]),
                  runge_Kutta_K_Mecha_Sensor_Q(3 * metaParam->numCellsMax[0]),
                  runge_Kutta_InitPos(metaParam->numCellsMax[0]),
                  cellCandidateAxes(NUMAXESmax * metaParam->numCellsMax[0]),
                  cellCandidateAxesUpdate(NUMAXESmax * metaParam->numCellsMax[0]),
                  cellLigandUpdate(NUMLIGmax * metaParam->numCellsMax[0]),
                  cellShapeRatio(metaParam->numCellsMax[0]),
                  cellSurface(metaParam->numCellsMax[0]),
                  cellContactSurfaceArea(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellEquilibriumDistance(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellNeighbIsLateral(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellAttractionCoefficient(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellRepulsionCoefficient(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellPlanarRigidityCoefficient(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellIntercalateWithNeighb(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellIntercalationIntensity(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellIntercalationBipolar(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellApicalConstrictionWithNeighb(NUMNEIGHBTOPOmax * metaParam->numCellsMax[0]),
                  cellBlebbingMode(metaParam->numCellsMax[0]),
                  cellRandomBlebbingAxis(metaParam->numCellsMax[0]),
                  cellEpiId(metaParam->numCellsMax[0]),
                  cellProtein(NUMPROTEINmax * metaParam->numCellsMax[0]),
                  cellProteinUpdate(NUMPROTEINmax * metaParam->numCellsMax[0])
                  ,
                  customState()
              { }

            /** State class destructor. */
            ~State() throw () {}

        protected:

            /** Is used to store the adress of the host MetaParam object that is used to dimension the thrust vectors. */
            MetaParam<HOST>* metaParam;

        public:

            /******************************************
            /****  Real State (saved as xml files) ****
            /******************************************

            /** Is used to store the current number of cells. */
            typename Backend<T>::vecUint          numCells;

            typename Backend<T>::vecUint          currentTimeStep;

            /** Is used to store the spatial position of the cells. */
            typename Backend<T>::vecD3            cellPosition;


            typename Backend<T>::vecD3            embryoCenter;
            //typename Backend<T>::vecD3            embryoAxes;

            /** Is used to store the second axis of the cells. */
            typename Backend<T>::vecD3            cellAxisAB;

            /** Is used to store the spatial position of the cells. */
            typename Backend<T>::vecD3            cellRadius;

            /** Is used to store the cell cycle state.*/ 
            typename Backend<T>::vecUint          cellState;

            /** Is used to store the previous mitotis time.*/ 
            typename Backend<T>::vecUint          cellTimer;

            /** Is used to store the cell generation.*/ 
            typename Backend<T>::vecUint          cellGeneration;

            /** Is used to store the current life length of the cell from its previous mitosis time.*/ 
            typename Backend<T>::vecUint          cellCycleLength;

            /** Is used to store the binary identity of each cell. 011101 means daughter 1 of daughter 0 of daughter 1 of daughter 1 of daughter 1 of zygote.*/
            typename Backend<T>::vecUint          cellId_bits;

            typename Backend<T>::vecDouble        cellLigand;

            // should not be in state but must be initialized because of algorithm 
            typename Backend<T>::vecUint          cellEpiIsPolarized;
            typename Backend<T>::vecD3            cellProtrusionExtForce;
            typename Backend<T>::vecDouble        cellMechanoSensorQ;


            /*********************************************/
            /********  Must be initialized *********/
            /*********************************************/

            /** Is used to store the read-index in the gaussian random number vector.*/ 
            typename Backend<T>::vecUint          randomGaussian_Counter;

            /** Is used to store the read-index in the uniform random number vector.*/ 
            typename Backend<T>::vecUint          randomUniform_Counter;


            typename Backend<T>::vecUint          errorCode;

            /*********************************************/
            /********  Algo                      *********/
            /*********************************************/

            typename Backend<T>::vecUint          cellType;

            /** Is used to store the first axis of the cells. */
            typename Backend<T>::vecD3            cellAxis1;

            /** Is used to store the side length of a grid box. This value must be larger than two times the
            * larger cell interaction distance.
            */
            typename Backend<T>::vecDouble        gridBoxSize;

            /** Is used to store the side length of the grid. */
            typename Backend<T>::vecDouble        worldSize;

            /** Is used to store the maximum cell position. */
            typename Backend<T>::vecDouble        worldMax;
           
            /** Is used to store the origin position of the grid. */
            typename Backend<T>::vecDouble        worldOrigin;

            /** Is used to store the number of boxes along an axis of the grid. */
            typename Backend<T>::vecUint          gridSize;

            /** Is used to store the number of particles in each box of the grid. */
            typename Backend<T>::vecUint          gridPartNum;

            /** Is used to store the ids of the particles in each box of the grid. */
            typename Backend<T>::vecUint          gridPartId;

            /** Is used to store the ids of the neighbor cells according to the metric criteria. */
            typename Backend<T>::vecUint          cellMetricNeighbId;
            
            /** Is used to store the number of neighbor cells according to the metric criteria. */
            typename Backend<T>::vecUint          cellMetricNeighbNum;

            /** Is used to store the ids of the neighbor cells according to the topological criteria. */
            typename Backend<T>::vecUint          cellTopologicalNeighbId;
            
            /** Is used to store the number of neighbor cells according to the topological criteria. */
            typename Backend<T>::vecUint          cellTopologicalNeighbNum;
            
            /** Is used to store the angle of the neigbhor cell according to the metric criteria. */
            typename Backend<T>::vecDouble        cellMetricNeighbAngle;

            /** Is used to store the forces exerted between the cells. */
            typename Backend<T>::vecD3            cellForces;
            typename Backend<T>::vecD3            cellProtrusionExtForces;
            typename Backend<T>::vecDouble        cellMechanoSensorQs;

            /** Is used to store the number of forces exerted between the cells. */
            typename Backend<T>::vecUint          cellForcesNum;
            typename Backend<T>::vecUint          cellProtrusionExtForcesNum;
            typename Backend<T>::vecUint          cellMechanoSensorQsNum;

            /** Is used to store the intermediate position used for the Runge Kutta integration. */
            typename Backend<T>::vecD3            runge_Kutta_K;
            typename Backend<T>::vecD3            runge_Kutta_K_Protr_Ext;
            typename Backend<T>::vecDouble        runge_Kutta_K_Mecha_Sensor_Q;

            typename Backend<T>::vecD3            runge_Kutta_InitPos;

            typename Backend<T>::vecD3            cellCandidateAxes;
            typename Backend<T>::vecD3            cellCandidateAxesUpdate;

            typename Backend<T>::vecDouble        cellLigandUpdate;

            typename Backend<T>::vecDouble        cellShapeRatio;
            typename Backend<T>::vecDouble        cellSurface;

            /********************************************
            ****** Mechanical state variable ************
            ********************************************/

            typename Backend<T>::vecDouble        cellContactSurfaceArea;
            typename Backend<T>::vecDouble        cellEquilibriumDistance;
            typename Backend<T>::vecUint          cellNeighbIsLateral;
            typename Backend<T>::vecDouble        cellAttractionCoefficient;
            typename Backend<T>::vecDouble        cellRepulsionCoefficient;
            typename Backend<T>::vecDouble        cellPlanarRigidityCoefficient;
            typename Backend<T>::vecUint          cellIntercalateWithNeighb;
            typename Backend<T>::vecDouble        cellIntercalationIntensity;
            typename Backend<T>::vecUint          cellIntercalationBipolar;
            typename Backend<T>::vecUint          cellApicalConstrictionWithNeighb;
            typename Backend<T>::vecUint          cellBlebbingMode;
            typename Backend<T>::vecD3            cellRandomBlebbingAxis;
            typename Backend<T>::vecUint          cellEpiId;

            /********************************************
            *************** GRN Specifics ***************
            ********************************************/

            typename Backend<T>::vecDouble            cellProtein;
            typename Backend<T>::vecDouble            cellProteinUpdate;

            /********************************************
            *************** Custom State ****************
            ********************************************/

            typename BackendCustom<T>::CustomState         customState;


            // Only copy initialized vectors
            template<typename StateType>
              State& copy(const StateType & other){
              
              numCells = other.numCells;
              currentTimeStep = other.currentTimeStep;
              cellPosition = other.cellPosition;
              embryoCenter = other.embryoCenter;
              cellAxisAB = other.cellAxisAB;
              cellRadius = other.cellRadius;
              cellState = other.cellState;
              cellTimer = other.cellTimer;
              cellGeneration = other.cellGeneration;
              cellCycleLength = other.cellCycleLength;
              cellId_bits = other.cellId_bits;
              cellLigand = other.cellLigand;
              cellEpiIsPolarized = other.cellEpiIsPolarized;
              cellProtrusionExtForce = other.cellProtrusionExtForce;
              cellMechanoSensorQ = other.cellMechanoSensorQ;
              randomGaussian_Counter = other.randomGaussian_Counter;
              randomUniform_Counter = other.randomUniform_Counter;
              errorCode = other.errorCode;
              cellProtein = other.cellProtein;
              cellProteinUpdate = other.cellProteinUpdate;
              
              customState.copy(other.customState);

              return *this;
            }

    };


}

#endif
