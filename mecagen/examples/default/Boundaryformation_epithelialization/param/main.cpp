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
    // param.cellCycleParams[0].mode = 0; //geometric progression
    // param.cellCycleParams[0].param1 = 1;
    // param.cellCycleParams[0].param2 = 1;
    // param.cellCycleParams[0].volume_ratio = .5;    
    // param.cellCycleParams[0].mPhaseLength = 10;    //10 ts equal 1 minute 
    param.cellCycleParams[0].mode = 2; //constant cycle
    param.cellCycleParams[0].param1 = 100000;
    param.cellCycleParams[0].volume_ratio = 1;   

    /***** MechaParams ****/
    for(uint i=0; i<3; i++){
      for(uint j=0; j<3; j++){
        param.mechaParams[0].maximumDistanceCoefficient[3*i+j]     = 1.6;//1.2414;
        param.mechaParams[0].surfaceScaling[3*i+j]                 = 1.3697;
        param.mechaParams[0].equilibriumDistanceCoefficient[3*i+j] = 0.9523128;
        param.mechaParams[0].repulsionCoefficient[3*i+j]           = 100.0;
        param.mechaParams[0].planarRigidityCoefficient[3*i+j]      = 4000.0;
      }
    }

    /*** Polarization AXes ***/
    param.numPolarizationAxes[0] = 2;

    // posterior border axis, following Delta
    param.polarizationAxisParams[0].idlig                   = 0; // Delta
    param.polarizationAxisParams[0].compMode                = 0; // weighted neighboring link
    param.polarizationAxisParams[0].apicoBasalInEpithelium  = 1; // defines the AB axis
    
    // anterior border axis, following Delto
    param.polarizationAxisParams[1].idlig                   = 1; // Delto
    param.polarizationAxisParams[1].compMode                = 0; // weighted neighboring link
    param.polarizationAxisParams[1].apicoBasalInEpithelium  = 1; // defines the AB axis

    /*** Ligands ***/
    param.numLigands[0] = 2;

    param.ligandParams[0].diffusion = .0;
    param.ligandParams[0].kappa =  0.02; 
    sprintf(param.ligandParams[0].name, "Delta-lig");

    param.ligandParams[1].diffusion = .0;
    param.ligandParams[1].kappa =  0.02; 
    sprintf(param.ligandParams[1].name, "Delto-lig");

    /**** Proteins *******/
    param.numProteins[0] = 11;
    param.proteins[0].kappa = .015;
    sprintf(param.proteins[0].name, "X");
    param.proteins[1].kappa = .015;
    sprintf(param.proteins[1].name, "Delta");
    param.proteins[2].kappa = .015;
    sprintf(param.proteins[2].name, "Delto");
    param.proteins[3].kappa = .015; //.99;
    sprintf(param.proteins[3].name, "Notch");
    param.proteins[4].kappa = .015; //.99;
    sprintf(param.proteins[4].name, "Epi-inducer");
    param.proteins[5].kappa = .015;
    sprintf(param.proteins[5].name, "Epi");
    param.proteins[6].kappa = .015;
    sprintf(param.proteins[6].name, "Epi-inducer-deg");
    param.proteins[7].kappa = .015;
    sprintf(param.proteins[7].name, "Epi2-inducer");
    param.proteins[8].kappa = .015;
    sprintf(param.proteins[8].name, "Epi2");
    param.proteins[9].kappa = .015;
    sprintf(param.proteins[9].name, "Epi2-inducer-deg");
    param.proteins[10].kappa = .015;
    sprintf(param.proteins[10].name, "Anterior");

    /*** ProteinNodes ****/
    param.numProteinNodes[0] = 2;
    
    // X
    param.proteinNodes[0].outputProteinID = 0;
    param.proteinNodes[0].Xmin = d3(-10000.0,-10000.0,-10000.0);
    param.proteinNodes[0].Xmax = d3(0.0,10000.0,10000.0);
    param.proteinNodes[0].tmin = 200;
    param.proteinNodes[0].tmax = 210;
    param.proteinNodes[0].quantity = 1.0;
    
    // Notch
    param.proteinNodes[1].outputProteinID = 3;
    param.proteinNodes[1].Xmin = d3(-10000.0,-10000.0,-10000.0);
    param.proteinNodes[1].Xmax = d3(10000.0,10000.0,10000.0);
    param.proteinNodes[1].tmin = 0;
    param.proteinNodes[1].tmax = 10000;
    param.proteinNodes[1].quantity = 1.0;

    /*** PPInteractions ****/
    param.numPPInteractions[0] = 0;//2;

    param.ppInteractions[0].numReactant         = 2;
    param.ppInteractions[0].reactantID[0]       = 4;  // Epi-inducer
    param.ppInteractions[0].x[0]                = 1;  
    param.ppInteractions[0].alpha[0]            = 4;    
    param.ppInteractions[0].reactantID[1]       = 5;  // Epi
    param.ppInteractions[0].x[1]                = 1;  
    param.ppInteractions[0].alpha[1]            = 0;  // no consumption
    param.ppInteractions[0].outputProteinID     = 6;  // Epi-inducer-deg
    param.ppInteractions[0].outputProteinAlpha  = 4;
    param.ppInteractions[0].k                   = .0002;

    param.ppInteractions[1].numReactant         = 2;
    param.ppInteractions[1].reactantID[0]       = 7;  // Epi_2-inducer
    param.ppInteractions[1].x[0]                = 1;  
    param.ppInteractions[1].alpha[0]            = 4;    
    param.ppInteractions[1].reactantID[1]       = 8;  // Epi_2
    param.ppInteractions[1].x[1]                = 1;  
    param.ppInteractions[1].alpha[1]            = 0;  // no consumption
    param.ppInteractions[1].outputProteinID     = 9;  // Epi_2-inducer-deg
    param.ppInteractions[1].outputProteinAlpha  = 4;
    param.ppInteractions[1].k                   = .0002;
    
    /*** Receptors ****/
    param.numReceptors[0] = 0;

    /*** TransReceptors ****/
    param.numTransReceptors[0] = 2;

    param.transReceptors[0].tau                = .001; 
    param.transReceptors[0].receptorProtID     = 3; // Notch
    param.transReceptors[0].ligID              = 0; // Delta_lig
    param.transReceptors[0].outputProtID       = 4; // Epi-inducer
    param.transReceptors[0].x_receptorProt     = 1;
    param.transReceptors[0].x_lig              = 1;
    param.transReceptors[0].alpha_lig          = 0;  //the ligand is not consumed 
    param.transReceptors[0].alpha_receptorProt = 0;  //the receptor is not consumed
    param.transReceptors[0].alpha_outputProt   = 1;

    param.transReceptors[1].tau                = .001; 
    param.transReceptors[1].receptorProtID     = 3; // Notch
    param.transReceptors[1].ligID              = 1; // Delto_lig
    param.transReceptors[1].outputProtID       = 7; // Epi_2-inducer
    param.transReceptors[1].x_receptorProt     = 1;
    param.transReceptors[1].x_lig              = 1;
    param.transReceptors[1].alpha_lig          = 0;  //the ligand is not consumed 
    param.transReceptors[1].alpha_receptorProt = 0;  //the receptor is not consumed
    param.transReceptors[1].alpha_outputProt   = 1;

    /*** Secretors ***/
    param.numSecretors[0] = 2;

    //Delta -> Delta_lig
    param.secretors[0].outputLigandID = 0;
    param.secretors[0].inputProteinID = 1;
    param.secretors[0].sigma = .015;

    //Delto -> Delto_lig
    param.secretors[1].outputLigandID = 1;
    param.secretors[1].inputProteinID = 2;
    param.secretors[1].sigma = .015;

    /*** Genes ****/
    param.numGenes[0] = 5;

    // Delta
    param.genes[0].outputProteinID = 1;
    param.genes[0].gamma = 1.5;
    param.genes[0].regEl.logicalFunction = 0; // AND function
    param.genes[0].regEl.numInputProtein = 1; 
    param.genes[0].regEl.inputProteinID[0] = 10;   // Anterior
    param.genes[0].regEl.inputThreshold[0] = 30.0;
    param.genes[0].regEl.inputType[0]      = 1;

    // Epi
    param.genes[1].outputProteinID = 5;
    param.genes[1].gamma = 1.5;
    param.genes[1].regEl.logicalFunction = 3; // (Epi-inducer AND X) OR (Epi)
    param.genes[1].regEl.numInputProtein = 3;   
    param.genes[1].regEl.inputProteinID[0] = 4;   // Epi-inducer
    param.genes[1].regEl.inputThreshold[0] = 10.0;
    param.genes[1].regEl.inputType[0]      = 1;
    param.genes[1].regEl.inputProteinID[1] = 10;   // Anterior
    param.genes[1].regEl.inputThreshold[1] = 1.0;
    param.genes[1].regEl.inputType[1]      = 0;
    param.genes[1].regEl.inputProteinID[2] = 5;   // Epi (auto-activation)
    param.genes[1].regEl.inputThreshold[2] = 10.0;
    param.genes[1].regEl.inputType[2]      = 1;

    // Delto
    param.genes[2].outputProteinID = 2;
    param.genes[2].gamma = 1.5;
    param.genes[2].regEl.logicalFunction = 0; // AND function
    param.genes[2].regEl.numInputProtein = 1; 
    param.genes[2].regEl.inputProteinID[0] = 5;   // Epi
    param.genes[2].regEl.inputThreshold[0] = 90.0;
    param.genes[2].regEl.inputType[0]      = 1;    

    // Epi_2
    param.genes[3].outputProteinID = 8;
    param.genes[3].gamma = 1.5;
    param.genes[3].regEl.logicalFunction = 3; // (Epi-inducer AND X) OR (Epi)
    param.genes[3].regEl.numInputProtein = 3;   
    param.genes[3].regEl.inputProteinID[0] = 7;   // Epi_2-inducer
    param.genes[3].regEl.inputThreshold[0] = 10.0;
    param.genes[3].regEl.inputType[0]      = 1;
    param.genes[3].regEl.inputProteinID[1] = 10;   // Anterior
    param.genes[3].regEl.inputThreshold[1] = 1.0;
    param.genes[3].regEl.inputType[1]      = 1;
    param.genes[3].regEl.inputProteinID[2] = 8;   // Epi_2 (auto-activation)
    param.genes[3].regEl.inputThreshold[2] = 10.0;
    param.genes[3].regEl.inputType[2]      = 1;    
    // param.genes[3].regEl.logicalFunction = 0; // (Epi-inducer AND X) OR (Epi)
    // param.genes[3].regEl.numInputProtein = 2;   
    // param.genes[3].regEl.inputProteinID[0] = 7;   // Epi_2-inducer
    // param.genes[3].regEl.inputThreshold[0] = 10.0;
    // param.genes[3].regEl.inputType[0]      = 1;
    // param.genes[3].regEl.inputProteinID[1] = 10;   // Anterior
    // param.genes[3].regEl.inputThreshold[1] = 1.0;
    // param.genes[3].regEl.inputType[1]      = 1;


    // Anterior
    param.genes[4].outputProteinID = 10;
    param.genes[4].gamma = 1.5;
    param.genes[4].regEl.logicalFunction = 1; // OR
    param.genes[4].regEl.numInputProtein = 2;   
    param.genes[4].regEl.inputProteinID[0] = 10;   // Anterior 
    param.genes[4].regEl.inputThreshold[0] = 10.0;
    param.genes[4].regEl.inputType[0]      = 1;
    param.genes[4].regEl.inputProteinID[1] = 0;   // X
    param.genes[4].regEl.inputThreshold[1] = 1.0;
    param.genes[4].regEl.inputType[1]      = 1;

    /*** Polarization Nodes ***/
    param.numPolarizationNodes[0] = 2;

    param.polarizationNodes[0].axisID = 0; // Delta polarization axis
    param.polarizationNodes[0].regEl.logicalFunction = 0; // AND function
    param.polarizationNodes[0].regEl.numInputProtein = 2;   
    param.polarizationNodes[0].regEl.inputProteinID[0] = 4;   // Epi-inducer
    param.polarizationNodes[0].regEl.inputThreshold[0] = 5.0;
    param.polarizationNodes[0].regEl.inputType[0]      = 1;
    param.polarizationNodes[0].regEl.inputProteinID[1] = 10;   // Anterior
    param.polarizationNodes[0].regEl.inputThreshold[1] = 1.0;
    param.polarizationNodes[0].regEl.inputType[1]      = 0;

    param.polarizationNodes[1].axisID = 1; // Delto polarization axis
    param.polarizationNodes[1].regEl.logicalFunction = 0; // AND function
    param.polarizationNodes[1].regEl.numInputProtein = 2;   
    param.polarizationNodes[1].regEl.inputProteinID[0] = 7;   // Epi_2-inducer
    param.polarizationNodes[1].regEl.inputThreshold[0] = 5.0;
    param.polarizationNodes[1].regEl.inputType[0]      = 1;
    param.polarizationNodes[1].regEl.inputProteinID[1] = 10;   // Anterior
    param.polarizationNodes[1].regEl.inputThreshold[1] = 1.0;
    param.polarizationNodes[1].regEl.inputType[1]      = 1;

    /*** EpiPolarization Nodes ***/
    param.numEpiPolarizationNodes[0] = 2;

    param.epiPolarizationNodes[0] = 0;
    param.epiPolarizationNodes[1] = 1;
    
    /*** Adhesion Nodes ***/
    param.numAdhesionNodes[0] = 1;
    param.adhesionNodes[0].mode = 3;        // lazy mode: constant value, independent from protein concentration
    param.adhesionNodes[0].k_adh = 100.0;
    param.adhesionNodes[0].proteinID = 0;   // unused in lazy mode

    /*** Celltype Nodes ***/
    param.cellTypeNodes[0].numInputProtein = 0; // Mesenchymal cells

    param.cellTypeNodes[1].numInputProtein   = 2; // Epithelial cells
    param.cellTypeNodes[1].logicalFunction   = 1;     // OR //0; //and 
    param.cellTypeNodes[1].inputProteinID[0] = 5;   // Epi
    param.cellTypeNodes[1].inputThreshold[0] = 70.0;
    param.cellTypeNodes[1].inputType[0]      = 1;
    param.cellTypeNodes[1].inputProteinID[1] = 8;   // Epi_2
    param.cellTypeNodes[1].inputThreshold[1] = 70.0;
    param.cellTypeNodes[1].inputType[1]      = 1;
    
    /***** Display GRN specifications ****/
    param.display();
    
    //save state as xml file
    save< Param_Host >(param, "param_archive.xml");
  }
