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

#ifndef _CUSTOMOBJECTS_
#define _CUSTOMOBJECTS_

#include "thrust_objects.hpp"
#include "define.hpp"

#define NUMCUSTOMCELLmax 10000
#define NUMPARTYOLKINTERIOR 500
#define NUMPARTYOLKMEMBRANE 2562
#define NUMPARTYOLK (NUMPARTYOLKINTERIOR+NUMPARTYOLKMEMBRANE)
#define NUMNEIGHMYMYmax 18
#define YOLKGRIDSIZEmax 16
#define YOLKGRIDNUMPARTmax 300
#define NUMPARTEVLmax 3000
#define EVLGRIDSIZEmax 16
#define EVLGRIDNUMPARTmax 200
#define CELLSYOLKGRIDSIZEmax 32
#define CELLSYOLKGRIDNUMPARTmax 200
#define NUMLATITUDE 24

namespace mg {

  struct CustomParams{

    double  yolkInteriorCmax;
    double  yolkInteriorSurfaceScaling;
    double  yolkInteriorAttractionCoefficient;
    double  yolkInteriorRepulsionCoefficient;
    double  yolkMembraneStiffness;
    double  yolkMembraneRLCoeff;
    double  evlRLCoeff;
    double  evlCmax;
    double  evlStiffness;
    double  evlRadiusAB;
    double  cellsYolkCmax;
    double  cellsYolkSurfaceScaling;
    double  cellsYolkEquilibriumDistance;
    double  cellsYolkAttractionCoefficient;
    double  cellsYolkRepulsionCoefficient;
    double  cellsEvlCmax;
    double  cellsEvlSurfaceScaling;
    double  cellsEvlEquilibriumDistance;
    double  cellsEvlAttractionCoefficient;
    double  cellsEvlRepulsionCoefficient;
    double  yolkMarginEvlStiffness;
    double  marginResistance;
    double  evlGrowthThreshold;
    double  evlLateralGrowthRatio;
    double  evlRadiusLimit;
    uint    yolkLigandId;
    uint    evlLigandId;
    double  yolkLigandUpdate;
    double  evlLigandUpdate;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
        ar
          & BOOST_SERIALIZATION_NVP(yolkInteriorCmax)
          & BOOST_SERIALIZATION_NVP(yolkInteriorSurfaceScaling)
          & BOOST_SERIALIZATION_NVP(yolkInteriorAttractionCoefficient)
          & BOOST_SERIALIZATION_NVP(yolkInteriorRepulsionCoefficient)
          & BOOST_SERIALIZATION_NVP(yolkMembraneStiffness)
          & BOOST_SERIALIZATION_NVP(yolkMembraneRLCoeff)
          & BOOST_SERIALIZATION_NVP(evlRLCoeff)
          & BOOST_SERIALIZATION_NVP(evlCmax)
          & BOOST_SERIALIZATION_NVP(evlStiffness)
          & BOOST_SERIALIZATION_NVP(evlRadiusAB)
          & BOOST_SERIALIZATION_NVP(cellsYolkCmax)
          & BOOST_SERIALIZATION_NVP(cellsYolkSurfaceScaling)
          & BOOST_SERIALIZATION_NVP(cellsYolkEquilibriumDistance)
          & BOOST_SERIALIZATION_NVP(cellsYolkAttractionCoefficient)
          & BOOST_SERIALIZATION_NVP(cellsYolkRepulsionCoefficient)
          & BOOST_SERIALIZATION_NVP(cellsEvlCmax)
          & BOOST_SERIALIZATION_NVP(cellsEvlSurfaceScaling)
          & BOOST_SERIALIZATION_NVP(cellsEvlEquilibriumDistance)
          & BOOST_SERIALIZATION_NVP(cellsEvlAttractionCoefficient)
          & BOOST_SERIALIZATION_NVP(cellsEvlRepulsionCoefficient)
          & BOOST_SERIALIZATION_NVP(yolkMarginEvlStiffness)
          & BOOST_SERIALIZATION_NVP(marginResistance)
          & BOOST_SERIALIZATION_NVP(evlGrowthThreshold)
          & BOOST_SERIALIZATION_NVP(evlLateralGrowthRatio)
          & BOOST_SERIALIZATION_NVP(evlRadiusLimit)
          & BOOST_SERIALIZATION_NVP(yolkLigandId)
          & BOOST_SERIALIZATION_NVP(evlLigandId)
          & BOOST_SERIALIZATION_NVP(yolkLigandUpdate)
          & BOOST_SERIALIZATION_NVP(evlLigandUpdate)
          ;
      }
  };

  template< int T >
  struct CustomStateTemplate{

    // State members. Must be initialized
    typename Backend<T>::vecD3        yolkPosition;
    typename Backend<T>::vecUint      yolkMembraneNeighbNum;
    typename Backend<T>::vecUint      yolkMembraneNeighbId;
    typename Backend<T>::vecDouble    yolkMembraneNeighbRL;
    typename Backend<T>::vecDouble    yolkMembraneRadius;
    typename Backend<T>::vecDouble    yolkInteriorRadius;
    typename Backend<T>::vecDouble    yolkInteriorDistMax;
    typename Backend<T>::vecDouble    yolkInteriorDistEq;
    typename Backend<T>::vecUint      numPartEVL;
    typename Backend<T>::vecD3        evlPosition;
    typename Backend<T>::vecD3        evlRadius;
    typename Backend<T>::vecD3        evlNormal;
    typename Backend<T>::vecUint      evlTimer;
    typename Backend<T>::vecUint      yolkMembraneActivated;
    typename Backend<T>::vecUint      yolkMembraneEYSL;
    typename Backend<T>::vecD4        yolkMembraneTangentParams;
    typename Backend<T>::vecUint      yolkMembraneNextNum;
    typename Backend<T>::vecUint      yolkMembraneNextId;

    // Algo members. Do not require initialization
    typename Backend<T>::vecUint      yolkGridPartNum;
    typename Backend<T>::vecUint      yolkGridPartId;
    typename Backend<T>::vecUint      yolkInteriorMetricNeighbNum;
    typename Backend<T>::vecUint      yolkInteriorMetricNeighbId;
    typename Backend<T>::vecUint      yolkInteriorTopologicalNeighbNum;
    typename Backend<T>::vecUint      yolkInteriorTopologicalNeighbId;
    typename Backend<T>::vecD3        yolkForces;
    typename Backend<T>::vecUint      yolkForcesNum;
    typename Backend<T>::vecD3        yolk_Runge_Kutta_K;
    typename Backend<T>::vecD3        yolk_Runge_Kutta_InitPos;
    typename Backend<T>::vecUint      evlGridPartNum;
    typename Backend<T>::vecUint      evlGridPartId;
    typename Backend<T>::vecUint      evlMetricNeighbNum;
    typename Backend<T>::vecUint      evlMetricNeighbId;
    typename Backend<T>::vecUint      evlTopologicalNeighbNum;
    typename Backend<T>::vecUint      evlTopologicalNeighbId;
    typename Backend<T>::vecD3        evlForces;
    typename Backend<T>::vecUint      evlForcesNum;
    typename Backend<T>::vecD3        evl_Runge_Kutta_K;
    typename Backend<T>::vecD3        evl_Runge_Kutta_InitPos;
    typename Backend<T>::vecUint      cellsYolkGridPartNum;
    typename Backend<T>::vecUint      cellsYolkGridPartId;
    typename Backend<T>::vecUint      cellsYolkNeighbNum;
    typename Backend<T>::vecUint      yolkCellsNeighbNum;
    typename Backend<T>::vecUint      yolkCellsNeighbId;
    typename Backend<T>::vecUint      cellsEvlNeighbNum;
    typename Backend<T>::vecUint      cellsEvlNeighbId;
    typename Backend<T>::vecUint      yolkMarginEvlMetricNeighbNum;
    typename Backend<T>::vecUint      yolkMarginEvlMetricNeighbId;
    typename Backend<T>::vecUint      yolkMarginEvlTopologicalNeighbNum;
    typename Backend<T>::vecUint      yolkMarginEvlTopologicalNeighbId;
    typename Backend<T>::vecUint      yolkMembraneEYSLupdate;
    typename Backend<T>::vecDouble    evlPressure;

    // Constructor
    CustomStateTemplate(): 

                yolkPosition(NUMPARTYOLK),
                yolkMembraneNeighbNum(2*NUMPARTYOLKMEMBRANE),
                yolkMembraneNeighbId(NUMNEIGHMYMYmax*NUMPARTYOLKMEMBRANE),
                yolkMembraneNeighbRL(NUMNEIGHMYMYmax*NUMPARTYOLKMEMBRANE),
                yolkMembraneRadius(1),
                yolkInteriorRadius(1),
                yolkInteriorDistMax(1),
                yolkInteriorDistEq(1),
                numPartEVL(1),
                evlPosition(NUMPARTEVLmax),
                evlRadius(NUMPARTEVLmax),
                evlNormal(NUMPARTEVLmax),
                evlTimer(NUMPARTEVLmax),
                yolkMembraneActivated(NUMPARTYOLKMEMBRANE),
                yolkMembraneEYSL(NUMPARTYOLKMEMBRANE),
                yolkMembraneTangentParams(NUMPARTYOLKMEMBRANE),
                yolkMembraneNextNum(2 * NUMPARTYOLKMEMBRANE),
                yolkMembraneNextId(40 * NUMPARTYOLKMEMBRANE),

                yolkGridPartNum(YOLKGRIDSIZEmax*YOLKGRIDSIZEmax*YOLKGRIDSIZEmax),
                yolkGridPartId(YOLKGRIDNUMPARTmax*YOLKGRIDSIZEmax*YOLKGRIDSIZEmax*YOLKGRIDSIZEmax),
                yolkInteriorMetricNeighbNum(NUMPARTYOLKINTERIOR),
                yolkInteriorMetricNeighbId(NUMNEIGHBMETRICmax * NUMPARTYOLKINTERIOR),
                yolkInteriorTopologicalNeighbNum(NUMPARTYOLKINTERIOR),
                yolkInteriorTopologicalNeighbId(NUMNEIGHBTOPOmax * NUMPARTYOLKINTERIOR),
                yolkForces(NUMPARTYOLK * NUMFORCEmax),
                yolkForcesNum(NUMPARTYOLK),
                yolk_Runge_Kutta_K(3*NUMPARTYOLK),
                yolk_Runge_Kutta_InitPos(NUMPARTYOLK),
                evlGridPartNum(EVLGRIDSIZEmax*EVLGRIDSIZEmax*EVLGRIDSIZEmax),
                evlGridPartId(EVLGRIDNUMPARTmax*EVLGRIDSIZEmax*EVLGRIDSIZEmax*EVLGRIDSIZEmax),
                evlMetricNeighbNum(NUMPARTEVLmax),
                evlMetricNeighbId(NUMPARTEVLmax*NUMNEIGHBMETRICmax),
                evlTopologicalNeighbNum(NUMPARTEVLmax),
                evlTopologicalNeighbId(NUMNEIGHBTOPOmax * NUMPARTEVLmax),
                evlForces(NUMPARTEVLmax * NUMFORCEmax),
                evlForcesNum(NUMPARTEVLmax),
                evl_Runge_Kutta_K(3*NUMPARTEVLmax),
                evl_Runge_Kutta_InitPos(NUMPARTEVLmax),
                cellsYolkGridPartNum(CELLSYOLKGRIDSIZEmax*CELLSYOLKGRIDSIZEmax*CELLSYOLKGRIDSIZEmax),
                cellsYolkGridPartId(CELLSYOLKGRIDNUMPARTmax*CELLSYOLKGRIDSIZEmax*CELLSYOLKGRIDSIZEmax*CELLSYOLKGRIDSIZEmax),
                cellsYolkNeighbNum(NUMCUSTOMCELLmax),
                yolkCellsNeighbNum(NUMPARTYOLKMEMBRANE),
                yolkCellsNeighbId(NUMPARTYOLKMEMBRANE),
                cellsEvlNeighbNum(NUMCUSTOMCELLmax),
                cellsEvlNeighbId(NUMCUSTOMCELLmax),
                yolkMarginEvlMetricNeighbNum(NUMPARTYOLKMEMBRANE),
                yolkMarginEvlMetricNeighbId(NUMPARTYOLKMEMBRANE*NUMNEIGHBMETRICmax),
                yolkMarginEvlTopologicalNeighbNum(NUMPARTYOLKMEMBRANE),
                yolkMarginEvlTopologicalNeighbId(NUMPARTYOLKMEMBRANE*NUMNEIGHBTOPOmax),
                yolkMembraneEYSLupdate(NUMPARTYOLKMEMBRANE),
                evlPressure(NUMPARTEVLmax)
                {}

    CustomStateTemplate& copy(const CustomStateTemplate<HOST> & other){
      yolkPosition = other.yolkPosition;
      yolkMembraneNeighbNum = other.yolkMembraneNeighbNum;
      yolkMembraneNeighbId = other.yolkMembraneNeighbId;
      yolkMembraneNeighbRL = other.yolkMembraneNeighbRL;
      yolkMembraneRadius = other.yolkMembraneRadius;
      yolkInteriorRadius = other.yolkInteriorRadius;
      yolkInteriorDistMax = other.yolkInteriorDistMax;
      yolkInteriorDistEq = other.yolkInteriorDistEq;
      numPartEVL = other.numPartEVL;
      evlPosition = other.evlPosition;
      evlRadius = other.evlRadius;
      evlNormal = other.evlNormal;
      evlTimer = other.evlTimer;
      yolkMembraneActivated = other.yolkMembraneActivated;
      yolkMembraneEYSL = other.yolkMembraneEYSL;
      yolkMembraneTangentParams = other.yolkMembraneTangentParams;
      yolkMembraneNextNum = other.yolkMembraneNextNum;
      yolkMembraneNextId = other.yolkMembraneNextId;
      return *this;
    }

    CustomStateTemplate& copy(const CustomStateTemplate<DEVICE> & other){
      yolkPosition = other.yolkPosition;
      yolkMembraneNeighbNum = other.yolkMembraneNeighbNum;
      yolkMembraneNeighbId = other.yolkMembraneNeighbId;
      yolkMembraneNeighbRL = other.yolkMembraneNeighbRL;
      yolkMembraneRadius = other.yolkMembraneRadius;
      yolkInteriorRadius = other.yolkInteriorRadius;
      yolkInteriorDistMax = other.yolkInteriorDistMax;
      yolkInteriorDistEq = other.yolkInteriorDistEq;
      numPartEVL = other.numPartEVL;
      evlPosition = other.evlPosition;
      evlRadius = other.evlRadius;
      evlNormal = other.evlNormal;
      evlTimer = other.evlTimer;
      yolkMembraneActivated = other.yolkMembraneActivated;
      yolkMembraneEYSL = other.yolkMembraneEYSL;
      yolkMembraneTangentParams = other.yolkMembraneTangentParams;
      yolkMembraneNextNum = other.yolkMembraneNextNum;
      yolkMembraneNextId = other.yolkMembraneNextId;
      return *this;
    }

    // Serialization 
    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
        ar
          & BOOST_SERIALIZATION_NVP(yolkPosition)
          & BOOST_SERIALIZATION_NVP(yolkMembraneNeighbNum)
          & BOOST_SERIALIZATION_NVP(yolkMembraneNeighbId)
          & BOOST_SERIALIZATION_NVP(yolkMembraneNeighbRL)
          & BOOST_SERIALIZATION_NVP(yolkMembraneRadius)
          & BOOST_SERIALIZATION_NVP(yolkInteriorRadius)
          & BOOST_SERIALIZATION_NVP(yolkInteriorDistMax)
          & BOOST_SERIALIZATION_NVP(yolkInteriorDistEq)
          & BOOST_SERIALIZATION_NVP(numPartEVL)
          & BOOST_SERIALIZATION_NVP(evlPosition)
          & BOOST_SERIALIZATION_NVP(evlRadius)
          & BOOST_SERIALIZATION_NVP(evlNormal)
          & BOOST_SERIALIZATION_NVP(evlTimer)
          & BOOST_SERIALIZATION_NVP(yolkMembraneActivated)
          & BOOST_SERIALIZATION_NVP(yolkMembraneEYSL)
          & BOOST_SERIALIZATION_NVP(yolkMembraneTangentParams)
          & BOOST_SERIALIZATION_NVP(yolkMembraneNextNum)
          & BOOST_SERIALIZATION_NVP(yolkMembraneNextId);
      }
  };

  /** Backend template specialization mechanism.
   * According to the backend indicator, corresponding types of thrust vectors are defined with the same
   * denomination
   */
  template<int integer>
    struct BackendCustom;

  /** Device backend types specialization. */
  template<>
    struct BackendCustom<DEVICE>
    {
      typedef typename thrust::device_vector<CustomParams>            vecCustomParam;
      typedef CustomStateTemplate<DEVICE>                                CustomState;
    };

  /** Host backend types specialization. */
  template<>
    struct BackendCustom<HOST>
    {
      typedef typename thrust::host_vector<CustomParams>              vecCustomParam;
      typedef CustomStateTemplate<HOST>                                  CustomState;
    };

  struct CustomStateBuffer{

    d3      yolkPosition[NUMPARTYOLK];
    uint    yolkMembraneNeighbNum[2*NUMPARTYOLKMEMBRANE];
    uint    yolkMembraneNeighbId[NUMNEIGHMYMYmax*NUMPARTYOLKMEMBRANE];
    uint    yolkInteriorTopologicalNeighbNum[NUMPARTYOLKINTERIOR];
    uint    yolkInteriorTopologicalNeighbId[NUMPARTYOLKINTERIOR*NUMNEIGHBTOPOmax];
    uint    numPartEVL[1];
    d3      evlPosition[NUMPARTEVLmax];
    d3      evlNormal[NUMPARTEVLmax];
    uint    evlTopologicalNeighbNum[NUMPARTEVLmax];
    uint    evlTopologicalNeighbId[NUMPARTEVLmax*NUMNEIGHBTOPOmax];
    uint    yolkCellsNeighbNum[NUMPARTYOLKMEMBRANE];
    uint    yolkCellsNeighbId[NUMPARTYOLKMEMBRANE];
    uint    yolkMembraneActivated[NUMPARTYOLKMEMBRANE];
    uint    yolkMembraneEYSL[NUMPARTYOLKMEMBRANE];
    uint    cellsEvlNeighbNum[NUMCUSTOMCELLmax];
    uint    cellsEvlNeighbId[NUMCUSTOMCELLmax];
    uint    yolkMarginEvlTopologicalNeighbNum[NUMPARTYOLKMEMBRANE];
    uint    yolkMarginEvlTopologicalNeighbId[NUMPARTYOLKMEMBRANE*NUMNEIGHBMETRICmax];
    d4      yolkMembraneTangentParams[NUMPARTYOLKMEMBRANE];

    uint    yolkMembraneNextNum[2 * NUMPARTYOLKMEMBRANE];
    uint    yolkMembraneNextId[40 * NUMPARTYOLKMEMBRANE];

    double  yolkInteriorRadius[1];
    double  yolkMembraneRadius[1];

    void copy(BackendCustom<DEVICE>::CustomState *sca, uint numCells){
      thrust::copy(sca->yolkPosition.begin(), sca->yolkPosition.end(), yolkPosition);
      thrust::copy(sca->yolkMembraneNeighbNum.begin(), sca->yolkMembraneNeighbNum.end(), yolkMembraneNeighbNum);
      thrust::copy(sca->yolkMembraneNeighbId.begin(), sca->yolkMembraneNeighbId.end(), yolkMembraneNeighbId);
      thrust::copy(sca->yolkInteriorTopologicalNeighbNum.begin(), sca->yolkInteriorTopologicalNeighbNum.end(), yolkInteriorTopologicalNeighbNum);
      thrust::copy(sca->yolkInteriorTopologicalNeighbId.begin(), sca->yolkInteriorTopologicalNeighbId.end(), yolkInteriorTopologicalNeighbId);
      thrust::copy(sca->numPartEVL.begin(), sca->numPartEVL.end(), numPartEVL);
      thrust::copy(sca->evlPosition.begin(), sca->evlPosition.end(), evlPosition);
      thrust::copy(sca->evlNormal.begin(), sca->evlNormal.end(), evlNormal);
      thrust::copy(sca->evlTopologicalNeighbNum.begin(), sca->evlTopologicalNeighbNum.end(), evlTopologicalNeighbNum);
      thrust::copy(sca->evlTopologicalNeighbId.begin(), sca->evlTopologicalNeighbId.end(), evlTopologicalNeighbId);
      thrust::copy(sca->yolkCellsNeighbNum.begin(), sca->yolkCellsNeighbNum.end(), yolkCellsNeighbNum);
      thrust::copy(sca->yolkCellsNeighbId.begin(), sca->yolkCellsNeighbId.end(), yolkCellsNeighbId);
      thrust::copy(sca->yolkMembraneActivated.begin(), sca->yolkMembraneActivated.end(), yolkMembraneActivated);
      thrust::copy(sca->yolkMembraneEYSL.begin(), sca->yolkMembraneEYSL.end(), yolkMembraneEYSL);
      thrust::copy(sca->cellsEvlNeighbNum.begin(), sca->cellsEvlNeighbNum.end(), cellsEvlNeighbNum);
      thrust::copy(sca->cellsEvlNeighbId.begin(), sca->cellsEvlNeighbId.end(), cellsEvlNeighbId);
      thrust::copy(sca->yolkMarginEvlTopologicalNeighbNum.begin(), sca->yolkMarginEvlTopologicalNeighbNum.end(), yolkMarginEvlTopologicalNeighbNum);
      thrust::copy(sca->yolkMarginEvlTopologicalNeighbId.begin(), sca->yolkMarginEvlTopologicalNeighbId.end(), yolkMarginEvlTopologicalNeighbId);
      thrust::copy(sca->yolkMembraneTangentParams.begin(), sca->yolkMembraneTangentParams.end(),yolkMembraneTangentParams );
      
      thrust::copy(sca->yolkMembraneNextNum.begin(), sca->yolkMembraneNextNum.end(), yolkMembraneNextNum);
      thrust::copy(sca->yolkMembraneNextId.begin(), sca->yolkMembraneNextId.end(), yolkMembraneNextId);

      thrust::copy(sca->yolkInteriorRadius.begin(), sca->yolkInteriorRadius.end(), yolkInteriorRadius);
      thrust::copy(sca->yolkMembraneRadius.begin(), sca->yolkMembraneRadius.end(), yolkMembraneRadius);
      
    }
  };
}

BOOST_SAVE_THRUST_HOST_VECTOR(mg::CustomParams)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::CustomParams)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::CustomParams>)

#endif

