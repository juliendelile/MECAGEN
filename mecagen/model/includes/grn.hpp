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

#ifndef _GRN_H
#define _GRN_H

namespace mg {

  /** This function determines whether the cis-regulatory element is active or not.*/ 
  inline __host__ __device__ uint isActive(const RegulatoryElement* regel, const double *cellProteinAtIdx){

    if(regel->numInputProtein == 0){
      return 0;
    }

    uint TFbounded[4];
    for( uint j=0 ; j < regel->numInputProtein ; j++ ){
      if( cellProteinAtIdx[ regel->inputProteinID[j] ] > regel->inputThreshold[j] ){
        TFbounded[j] = 1;
      }
      else{
        TFbounded[j] = 0;
      }
    }

    uint active;
    switch (regel->logicalFunction)
    {
      case 0:
        // function type AND, all activators must be present and no repressor
        active=1;
        for(uint j=0 ; j<regel->numInputProtein ; j++){ 
          if( TFbounded[j] == 1 ){
            active = active && regel->inputType[j];
          }
          else {
            active = active && !regel->inputType[j];
          }
        }
        break;
      case 1:
        // function type OR, a single activator is sufficient 
        active=0;
        for(uint j=0 ; j<regel->numInputProtein ; j++){
          if( TFbounded[j] == 1 ){
            active = active || regel->inputType[j];
          }
          else {
            active = active || 0;
          }
        }
        break;
      case 2:
        active = CUSTOM_REGEL_FUNCTION_2;
        break;
      case 3:
        // Requires at least two TFs, the last must be an activator
        // function ( TF1 AND TF2 ... AND TFn-1) OR (TFn)
        // with TFn activator
        if( TFbounded[regel->numInputProtein-1] == 1 
              && regel->inputType[regel->numInputProtein-1] == 1
              ){
          return 1;
        }
        else{
          active=1;
          for(uint j=0 ; j<regel->numInputProtein-1 ; j++){ 
            if( TFbounded[j] == 1 ){
              active = active && regel->inputType[j];
            }
            else {
              active = active && !regel->inputType[j];
            }
          }
        }
        break;
      default:
         printf("unknown logicalFunction");
    }

    return active;
  }

  /** This functor calculates protein quantity updates according to GRN rules.*/
  struct computeGRN
  {
    double*                   cellProteinUpdate;  
    double*                   cellLigandUpdate;
    uint*                     errorCode;
    const double*             cellProtein;
    const double*             cellLigand;
    const uint                numProteins;
    const Protein*            proteins;        
    const uint                numGenes;
    const Gene*               genes;
    const uint                numReceptors;
    const Receptor*           receptors;
    const uint                numSecretors;
    const Secretor*           secretors;
    const uint                numPPInteractions;
    const PPInteraction*      ppInteractions;
    const uint                numProteinNodes;
    const ProteinNode*        proteinNodes;
    const MechanoSensorNode*  mechanoSensorNode;
    const double*             cellMechanoSensorQ;
    const d3*                 cellPosition;
    const uint                currentTimeStep;
    const double              deltaTime;
    const d3*                 embryoCenter;
    
    computeGRN(
                double*             _cellProteinUpdate,  
                double*             _cellLigandUpdate,
                uint*               _errorCode,
                double*             _cellProtein,
                double*             _cellLigand,
                uint                _numProteins,
                Protein*            _proteins,        
                uint                _numGenes,
                Gene*               _genes,
                uint                _numReceptors,
                Receptor*           _receptors,
                uint                _numSecretors,
                Secretor*           _secretors,
                uint                _numPPInteractions,
                PPInteraction*      _ppInteractions,
                uint                _numProteinNodes,
                ProteinNode*        _proteinNodes,
                MechanoSensorNode*  _mechanoSensorNode,
                double*             _cellMechanoSensorQ,
                d3*                 _cellPosition,
                uint                _currentTimeStep,
                double              _deltaTime,
                d3*                 _embryoCenter
            )
         :
            cellProteinUpdate(_cellProteinUpdate),
            cellLigandUpdate(_cellLigandUpdate),
            errorCode(_errorCode),
            cellProtein(_cellProtein),
            cellLigand(_cellLigand),
            numProteins(_numProteins),
            proteins(_proteins),
            numGenes(_numGenes),
            genes(_genes),
            numReceptors(_numReceptors),
            receptors(_receptors),
            numSecretors(_numSecretors),
            secretors(_secretors),
            numPPInteractions(_numPPInteractions),
            ppInteractions(_ppInteractions),
            numProteinNodes(_numProteinNodes),
            proteinNodes(_proteinNodes),
            mechanoSensorNode(_mechanoSensorNode),
            cellMechanoSensorQ(_cellMechanoSensorQ),
            cellPosition(_cellPosition),
            currentTimeStep(_currentTimeStep),
            deltaTime(_deltaTime),
            embryoCenter(_embryoCenter)
            {}

    __device__
    void operator()(const int& idx){
      
      //store cell idx protein in register memory
      double cellProtQ[2*NUMPROTEINmax];
      for(uint i=0;i<numProteins;i++){
        cellProtQ[i]                 = cellProtein[NUMPROTEINmax*idx + i];  //contains initial values
        cellProtQ[NUMPROTEINmax + i] = .0;                                  //contains update values
      }

      /*************************
      /*************************
      /*****    Genes     ******
      /*************************
      /************************/

      for(uint i=0;i<numGenes;i++){
        if( isActive(&(genes[i].regEl), &(cellProtQ[0])) ){
          cellProtQ[ NUMPROTEINmax + genes[i].outputProteinID ] += genes[i].beta * deltaTime;
        }
      }

      /*************************
      /*************************
      /*****   Receptors   *****
      /*************************
      /************************/

      for(uint i=0;i<numReceptors;i++){
        double r = receptors[i].tau 
                      * pow(cellProtQ[ receptors[i].receptorProtID ], receptors[i].x_receptorProt)
                      * pow(cellLigand[ NUMLIGmax * idx + receptors[i].ligID ], receptors[i].x_lig);
        r *= deltaTime;

        // Ligand update
        cellLigandUpdate[ NUMLIGmax + receptors[i].ligID ] -= receptors[i].alpha_lig * r;
        // Receptor protein update
        cellProtQ[ NUMPROTEINmax + receptors[i].receptorProtID ] -= receptors[i].alpha_receptorProt * r;
        // Transduced protein update
        cellProtQ[ NUMPROTEINmax + receptors[i].outputProtID ] += receptors[i].alpha_outputProt * r;
      }

      /*************************
      /*************************
      /*****   Secretors   *****
      /*************************
      /************************/
      
      for(uint i=0;i<numSecretors;i++){
        
        double updateQuantity = secretors[i].sigma * cellProtQ[ secretors[i].inputProteinID ] * deltaTime;

        if( updateQuantity > 0 ){
          cellProtQ[ NUMPROTEINmax + secretors[i].inputProteinID ] -= updateQuantity;
          cellLigandUpdate[NUMLIGmax * idx + secretors[i].outputLigandID ] += updateQuantity;
        }
      }

      /*************************
      /*************************
      /**** PPInteractions *****
      /*************************
      /************************/
      
      for(uint i=0;i<numPPInteractions;i++){
        double r = ppInteractions[i].k;
        for(uint j=0; j <  ppInteractions[i].numReactant; j++){
          uint reactantID = ppInteractions[i].reactantID[j];
          r *= pow(cellProtQ[ reactantID ], (double)ppInteractions[i].x[j]);
        }
        r *= deltaTime;
        cellProtQ[ NUMPROTEINmax + ppInteractions[i].outputProteinID ] += ppInteractions[i].outputProteinAlpha * r;
        for(uint j=0; j < ppInteractions[i].numReactant; j++){
          cellProtQ[ NUMPROTEINmax + ppInteractions[i].reactantID[j] ] -= ppInteractions[i].alpha[j] * r;
        }
      }   

      /*************************
      /*************************
      /***** Protein Nodes *****
      /*************************
      /************************/
      d3 pos = cellPosition[idx] - embryoCenter[0];

      for(uint i=0;i<numProteinNodes;i++){

        if( 
              currentTimeStep >= proteinNodes[i].tmin && currentTimeStep <= proteinNodes[i].tmax
          &&  pos.x >= proteinNodes[i].Xmin.x && pos.x <= proteinNodes[i].Xmax.x
          &&  pos.y >= proteinNodes[i].Xmin.y && pos.y <= proteinNodes[i].Xmax.y
          &&  pos.z >= proteinNodes[i].Xmin.z && pos.z <= proteinNodes[i].Xmax.z
          ){
          cellProtQ[ NUMPROTEINmax + proteinNodes[i].outputProteinID ] += proteinNodes[i].quantity * deltaTime;
        }
      }   

      /*************************
      /*************************
      /** MechanoTransduction **
      /*************************
      /************************/

      if( isActive(&(mechanoSensorNode[0].regEl), &(cellProtQ[0])) ){
        if( cellMechanoSensorQ[idx] > mechanoSensorNode[0].force_threshold ){
          cellProtQ[ NUMPROTEINmax + mechanoSensorNode[0].outputProteinID ] += mechanoSensorNode[0].xi * deltaTime;
        }
      }

      /*************************
      /*************************
      /*****  Degradation  *****
      /*************************
      /************************/

      for(uint i=0;i<numProteins;i++){
        cellProtQ[NUMPROTEINmax + i] -= proteins[i].kappa * cellProtQ[i] * deltaTime;
      }

      /*************************
      /*************************
      /*****  Store update *****
      /*************************
      /************************/

      for(uint i=0; i<numProteins; i++){       
        cellProteinUpdate[NUMPROTEINmax * idx + i] = cellProtQ[NUMPROTEINmax + i];
      }

    } // end operator()
  }; // end functor

  /** This functor calculates the protein quantity generated by ligand transduction. */
  struct transmembraneLigandSensing
  {
    const uint                numTransReceptors;
    const Receptor*           transReceptors;
    const uint*               cellTopologicalNeighbNum;
    const uint*               cellTopologicalNeighbId;
    const double*             cellSurface;
    const double*             cellContactSurfaceArea;
    const double*             cellProtein;
    double*                   cellProteinUpdate;
    const double*             cellLigand;
    const double              deltaTime;
    uint*                     errorCode;
    
    transmembraneLigandSensing(
                uint                _numTransReceptors,
                Receptor*           _transReceptors,
                uint*               _cellTopologicalNeighbNum,
                uint*               _cellTopologicalNeighbId,
                double*             _cellSurface,
                double*             _cellContactSurfaceArea,
                double*             _cellProtein,
                double*             _cellProteinUpdate,
                double*             _cellLigand,
                double              _deltaTime,
                uint*               _errorCode
            )
         :
            numTransReceptors(_numTransReceptors),
            transReceptors(_transReceptors),
            cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
            cellTopologicalNeighbId(_cellTopologicalNeighbId),
            cellSurface(_cellSurface),
            cellContactSurfaceArea(_cellContactSurfaceArea),
            cellProtein(_cellProtein),
            cellProteinUpdate(_cellProteinUpdate),
            cellLigand(_cellLigand),
            deltaTime(_deltaTime),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){
      
      if(numTransReceptors == 0){
        return;
      }

      //store cell idx protein in register memory
      double transQ[NUMTRANSRECEPTORmax];
      for(uint i=0;i<numTransReceptors;i++){
        transQ[i] = .0;
      }

      uint numNeighb = cellTopologicalNeighbNum[idx];
      double surface1 = cellSurface[idx];

      for(uint i=0; i<numNeighb; i++){

        uint topoNeighbIndex  = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId     = cellTopologicalNeighbId[topoNeighbIndex];

        double Aij = cellContactSurfaceArea[topoNeighbIndex];
        double surface2 = cellSurface[neighbCellId];

        for(uint j=0; j<numTransReceptors; j++){

          transQ[j] += transReceptors[j].tau 
                      * pow(Aij / surface1 * cellProtein[ NUMPROTEINmax * idx + transReceptors[j].receptorProtID ], transReceptors[j].x_receptorProt)
                      * pow(Aij / surface2 * cellLigand[ NUMLIGmax * neighbCellId + transReceptors[j].ligID ], transReceptors[j].x_lig);
        }
      }

      for(uint j=0; j < numTransReceptors; j++){
        // Transduced protein update
        cellProteinUpdate[ NUMPROTEINmax * idx + transReceptors[j].outputProtID ] += transReceptors[j].alpha_outputProt * transQ[j] * deltaTime;
      }
    }
  }; // end functor

  /** This functor integrates protein quantities update.*/
  struct updateProteins
  {
    const uint            numProteins;        
    const Protein*        proteins;
    double*               cellProtein;
    const double*         cellProteinUpdate;
    uint*                 errorCode;
    
    updateProteins(
                uint           _numProteins,        
                Protein*       _proteins,
                double*        _cellProtein,
                double*        _cellProteinUpdate,
                uint*          _errorCode
            )
         :
            numProteins(_numProteins),
            proteins(_proteins),
            cellProtein(_cellProtein),
            cellProteinUpdate(_cellProteinUpdate),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){
      
      for(uint i=0;i<numProteins;i++){
          
        double newQ = cellProtein[NUMPROTEINmax*idx + i] + cellProteinUpdate[NUMPROTEINmax*idx + i];
        cellProtein[NUMPROTEINmax*idx + i] = newQ;

        if( newQ < 0){
          // cellProtein[NUMPROTEINmax * idx + i] = .0;
          printf("Error: Protein %d in cell %d has a negative concentration. Please check your GRN specification. Probably, either a GRN constant or the simulation time step is too high.\n", i, idx);
          errorCode[0] = 9;
          return;
        }
      }
    }
  };

} //end namespace

#endif
