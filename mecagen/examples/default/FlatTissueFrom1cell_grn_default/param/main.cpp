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

#include "thrust_objects.hpp"
#include "param_host.hpp"
#include "serialization.hpp"

#include <stdio.h>

using namespace mg;

  int main( int argc, const char* argv[] ){
    
    Param_Host param;

    param.globalDamping[0] =      3000; 
    param.deltaTime[0] =          6.0;  //6 seconds per time step


    /**** Cell Cycle *****/
    param.cellCycleParams[0].mode = 0; //geometric progression
    param.cellCycleParams[0].param1 = 1;
    param.cellCycleParams[0].param2 = 1;
    param.cellCycleParams[0].volume_ratio = 1;    
    param.cellCycleParams[0].mPhaseLength = 1;       //10 ts equal 1 minute
    // param.cellCycleParams[0].mode = 2; //constant cycle
    // param.cellCycleParams[0].param1 = 1000;
    // param.cellCycleParams[0].volume_ratio = 1;   

    /***** MechaParams ****/
    for(uint i=0; i<3; i++){
      for(uint j=0; j<3; j++){
        param.mechaParams[0].maximumDistanceCoefficient[3*i+j]     = 1.6;//1.2414;
        param.mechaParams[0].surfaceScaling[3*i+j]                 = 1.3697;
        param.mechaParams[0].equilibriumDistanceCoefficient[3*i+j] = 0.9523128;
        param.mechaParams[0].repulsionCoefficient[3*i+j]           = 200.0;
        param.mechaParams[0].planarRigidityCoefficient[3*i+j]      = 100.0;
      }
    }


    /*** Polarization axes ***/
    param.numPolarizationAxes[0] = 0;

    /***********************/
    /**** GRN Specific *****/
    /***********************/

    /**** Proteins *******/
    param.numProteins[0] = 0;

    /*** ProteinNodes ****/
    param.numProteinNodes[0] = 0;

    /*** PPInteractions ****/
    param.numPPInteractions[0] = 0;
    
    /*** Receptors ****/
    param.numReceptors[0] = 0;

    /*** TransReceptors ****/
    param.numTransReceptors[0] = 0;

    /*** Secretors ***/
    param.numSecretors[0] = 0;

    /*** Genes ****/
    param.numGenes[0] = 0;

    /*** Polarization Nodes ****/
    param.numPolarizationNodes[0] = 0;

    /*** EpiPolarization Nodes ***/
    param.numEpiPolarizationNodes[0] = 0;
    
    /*** Adhesion Nodes ***/
    param.numAdhesionNodes[0] = 1;
    param.adhesionNodes[0].mode = 3;        // lazy mode: constant value, independent from protein concentration
    param.adhesionNodes[0].k_adh = 100.0;
    param.adhesionNodes[0].proteinID = 0;   // unused in lazy mode

    /*** Ligands ***/
    param.numLigands[0] = 0;

    /*** Celltypes Nodes ***/
    param.cellTypeNodes[0].numInputProtein = 0;
    param.cellTypeNodes[1].numInputProtein = 0;

    /***** Display GRN specifications ****/
    param.display();

    //save state as xml file
    save< Param_Host >(param, "param_archive.xml");
  }
