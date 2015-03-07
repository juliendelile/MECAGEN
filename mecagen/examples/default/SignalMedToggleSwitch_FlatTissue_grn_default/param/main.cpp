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
    param.cellCycleParams[0].volume_ratio = .5;    
    param.cellCycleParams[0].mPhaseLength = 10;    //10 ts equal 1 minute 
    // param.cellCycleParams[0].mode = 2; //constant cycle
    // param.cellCycleParams[0].param1 = 1000;
    // param.cellCycleParams[0].volume_ratio = 1;   

    /***** MechaParams ****/
    for(uint i=0; i<3; i++){
      for(uint j=0; j<3; j++){
        param.mechaParams[0].maximumDistanceCoefficient[3*i+j]     = 1.6;//1.2414;
        param.mechaParams[0].surfaceScaling[3*i+j]                 = 1.3697;
        param.mechaParams[0].equilibriumDistanceCoefficient[3*i+j] = 0.9523128;
        param.mechaParams[0].repulsionCoefficient[3*i+j]           = 100.0;
        param.mechaParams[0].planarRigidityCoefficient[3*i+j]      = 100.0;
      }
    }

    /*** Polarization AXes ***/
    param.numPolarizationAxes[0] = 0;
    
    /*** Ligands ***/
    param.numLigands[0] = 1;
    param.ligandParams[0].diffusion = .005; //0.0166667; //.1 / 6.0;
    param.ligandParams[0].chi =  0.02; //0.015166667; //.1 / 6.0; //.9
    sprintf(param.ligandParams[0].name, "Wnt");

    /**** Proteins *******/
    param.numProteins[0] = 11;
    param.proteins[0].kappa = .03 / 6.0; //.97;
    sprintf(param.proteins[0].name, "X");
    param.proteins[1].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[1].name, "Frizzled");
    param.proteins[2].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[2].name, "XIAP-inducer");
    param.proteins[3].kappa = .04 / 6.0; //.99;
    sprintf(param.proteins[3].name, "Beta-catenin");
    param.proteins[4].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[4].name, "XIAP");
    param.proteins[5].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[5].name, "Tcf");
    param.proteins[6].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[6].name, "Gro");
    param.proteins[7].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[7].name, "Tcf+");
    param.proteins[8].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[8].name, "Tcf-");
    param.proteins[9].kappa = .01 / 6.0; //.99;
    sprintf(param.proteins[9].name, "Target");
    param.proteins[10].kappa = .01 / 6.0; //.99;
    // sprintf(param.proteins[10].name, "Gro_Ubiquitinated");
    sprintf(param.proteins[10].name, "Gro^{Ubi}");


    /*** ProteinNodes ****/
    param.numProteinNodes[0] = 5;
    
    // X
    param.proteinNodes[0].outputProteinID = 0;
    param.proteinNodes[0].Xmin = d3(-10000.0,-10000.0,-10000.0);
    param.proteinNodes[0].Xmax = d3(0.0,0.0,10000.0);
    param.proteinNodes[0].tmin = 200;
    param.proteinNodes[0].tmax = 10000;
    param.proteinNodes[0].quantity = 1.3;
    
    // Frizzled
    param.proteinNodes[1].outputProteinID = 1;
    param.proteinNodes[1].Xmin = d3(-10000.0,-10000.0,-10000.0);
    param.proteinNodes[1].Xmax = d3(10000.0,10000.0,10000.0);
    param.proteinNodes[1].tmin = 0;
    param.proteinNodes[1].tmax = 10000;
    param.proteinNodes[1].quantity = .1;
    
    // XIAP-inducer
    param.proteinNodes[2].outputProteinID = 2;
    param.proteinNodes[2].Xmin = d3(-10000.0,-10000.0,-10000.0);
    param.proteinNodes[2].Xmax = d3(10000.0,10000.0,10000.0);
    param.proteinNodes[2].tmin = 0;
    param.proteinNodes[2].tmax = 10000;
    param.proteinNodes[2].quantity = .1;
    
    // Tcf
    param.proteinNodes[3].outputProteinID = 5;
    param.proteinNodes[3].Xmin = d3(-10000.0,-10000.0,-10000.0);
    param.proteinNodes[3].Xmax = d3(10000.0,10000.0,10000.0);
    param.proteinNodes[3].tmin = 0;
    param.proteinNodes[3].tmax = 10000;
    param.proteinNodes[3].quantity = .1;
    
    // Gro
    param.proteinNodes[4].outputProteinID = 6;
    param.proteinNodes[4].Xmin = d3(-10000.0,-10000.0,-10000.0);
    param.proteinNodes[4].Xmax = d3(10000.0,10000.0,10000.0);
    param.proteinNodes[4].tmin = 0;
    param.proteinNodes[4].tmax = 10000;
    param.proteinNodes[4].quantity = .1;

    /*** PPInteractions ****/
    param.numPPInteractions[0] = 3;

    param.ppInteractions[0].numReactant         = 2;
    param.ppInteractions[0].reactantID[0]       = 3;  // Beta-catenin
    param.ppInteractions[0].x[0]                = 1;  
    param.ppInteractions[0].alpha[0]            = 0;  // no consumption  
    param.ppInteractions[0].reactantID[1]       = 5;  // Tcf
    param.ppInteractions[0].x[1]                = 1;  
    param.ppInteractions[0].alpha[1]            = 0;  // no consumption
    param.ppInteractions[0].outputProteinID     = 7;  // Tcf+
    param.ppInteractions[0].outputProteinAlpha  = 1;
    param.ppInteractions[0].k                   = .00002;

    param.ppInteractions[1].numReactant         = 2;
    param.ppInteractions[1].reactantID[0]       = 6;  // Gro
    param.ppInteractions[1].x[0]                = 1;  
    param.ppInteractions[1].alpha[0]            = 0;  // no consumption  
    param.ppInteractions[1].reactantID[1]       = 5;  // Tcf
    param.ppInteractions[1].x[1]                = 1;  
    param.ppInteractions[1].alpha[1]            = 0;  // no consumption
    param.ppInteractions[1].outputProteinID     = 8;  // Tcf-
    param.ppInteractions[1].outputProteinAlpha  = 1;
    param.ppInteractions[1].k                   = .00002;

    param.ppInteractions[2].numReactant         = 2;
    param.ppInteractions[2].reactantID[0]       = 6;  // Gro
    param.ppInteractions[2].x[0]                = 1;  
    param.ppInteractions[2].alpha[0]            = 1;  // consumed  
    param.ppInteractions[2].reactantID[1]       = 4;  // XIAP
    param.ppInteractions[2].x[1]                = 1;  
    param.ppInteractions[2].alpha[1]            = 1;  
    param.ppInteractions[2].outputProteinID     = 10; // Gro_Ubiquitinated
    param.ppInteractions[2].outputProteinAlpha  = 1;
    param.ppInteractions[2].k                   = .00004;
    
    /*** Receptors ****/
    param.numReceptors[0] = 2;

    param.receptors[0].tau                = .0001; 
    param.receptors[0].receptorProtID     = 1;
    param.receptors[0].ligID              = 0;
    param.receptors[0].outputProtID       = 3;  // Beta-catenin
    param.receptors[0].x_receptorProt     = 1;
    param.receptors[0].x_lig              = 1;
    param.receptors[0].alpha_lig          = 0;  //the ligand is not consumed 
    param.receptors[0].alpha_receptorProt = 0;  //the receptor is not consumed
    param.receptors[0].alpha_outputProt   = 1;


    param.receptors[1].tau                = .00003; 
    param.receptors[1].receptorProtID     = 2;
    param.receptors[1].ligID              = 0;
    param.receptors[1].outputProtID       = 4;  // XIAP
    param.receptors[1].x_receptorProt     = 1;  
    param.receptors[1].x_lig              = 1;
    param.receptors[1].alpha_lig          = 0;  //the ligand is not consumed 
    param.receptors[1].alpha_receptorProt = 0;  //the receptor is not consumed
    param.receptors[1].alpha_outputProt   = 1;

    /*** TransReceptors ****/
    param.numTransReceptors[0] = 0;

    /*** Secretors ***/
    param.numSecretors[0] = 1;

    param.secretors[0].outputLigandID = 0;
    param.secretors[0].inputProteinID = 0;
    param.secretors[0].sigma = .1 / 6.0;

    /*** Genes ****/
    param.numGenes[0] = 1;

    // Target
    param.genes[0].outputProteinID = 9;
    param.genes[0].beta = .1;
    param.genes[0].regEl.logicalFunction = 0; // AND function
    param.genes[0].regEl.numInputProtein = 2; 
    param.genes[0].regEl.inputProteinID[0] = 7;   // Tcf+
    param.genes[0].regEl.inputThreshold[0] = 30.0;
    param.genes[0].regEl.inputType[0]      = 1;
    param.genes[0].regEl.inputProteinID[1] = 8;   // Tcf-
    param.genes[0].regEl.inputThreshold[1] = 30.0;
    param.genes[0].regEl.inputType[1]      = 0;

    /*** Polarization Nodes ***/
    param.numPolarizationNodes[0] = 0;

    /*** EpiPolarization Nodes ***/
    param.numEpiPolarizationNodes[0] = 0;
    
    /*** Adhesion Nodes ***/
    param.numAdhesionNodes[0] = 1;
    param.adhesionNodes[0].mode = 3;        // lazy mode: constant value, independent from protein concentration
    param.adhesionNodes[0].k_adh = 100.0;
    param.adhesionNodes[0].proteinID = 0;   // unused in lazy mode

    /*** Celltype Nodes ***/
    param.cellTypeNodes[0].numInputProtein = 0;
    param.cellTypeNodes[1].numInputProtein = 0;

    /***** Display GRN specifications ****/
    param.display();

    //save state as xml file
    save< Param_Host >(param, "param_archive.xml");
  }
