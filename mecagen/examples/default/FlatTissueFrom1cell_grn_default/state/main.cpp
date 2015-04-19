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

#include "metaparam.hpp"
#include "state_host.hpp"
#include "thrust_objects.hpp"
#include "serialization.hpp"

#include <stdio.h>

using namespace mg;

  int main( int argc, const char* argv[] ){
    
    MetaParam<HOST>     metaParam;
    
    // Maximum cell number allowed
    metaParam.numCellsMax[0] = 10000;

    // Here, we aim at populating a flat area with about R=880 cells of size R=10
    // hence, the estimated surface of the area should be about N * PI * R * R ~= 276460
    // i.e. approx. a square of side length 525
    // We squeeze the cells by shortening this length to 430

    metaParam.spatialBorderMin[0] = d3(-215,-215,-1);
    metaParam.spatialBorderMax[0] = d3(215,215,1);

    metaParam.displayScale[0] = 300.0;

    metaParam.grid_SizeMax[0]     = 64;

    /*** Embryo axes ***/
    metaParam.embryoAxes[0] = d3(1,0,0);
    metaParam.embryoAxes[1] = d3(0,1,0);
    metaParam.embryoAxes[2] = d3(0,0,1);

    save< MetaParam<HOST> >(metaParam, "metaparam_archive.xml");

    State_Host state(&metaParam);

    /*** Cell number ***/
    state.numCells[0] = 1; 

    /*** Timestep ***/
    state.currentTimeStep[0] = 0;   

    state.cellPosition[0] = d3(.0);
    state.embryoCenter[0] = d3(.0);

    /*** Cell radii ***/
    state.cellRadius[0] = d3(10,0,0);

    // /*** Cell apico-basal axes ***/
    // d3 avg_pos(0, 0, 0);

    // for(int i=0; i<state.numCells[0]; i++){
    //    avg_pos += state.cellPosition[i];
    // }

    // avg_pos /= (double)state.numCells[0];

    // d3 relPos;

    // for(int i=0; i<state.numCells[0]; i++){
    //   relPos = state.cellPosition[i] - avg_pos;
    //   state.cellAxisAB[i] = relPos / length(relPos);
    // }

    
    /*** Cell cycle ***/
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(100.0, 10.0);
    // std::normal_distribution<double> distribution(500.0, 100.0);

    for(uint i=0; i<state.numCells[0]; i++){
      state.cellState[i] = 0;
      state.cellTimer[i] = 0;
      state.cellGeneration[i] = 0;
      int ccl = (int)(distribution(generator));
      if(ccl<=0){
        // printf("Error lifetime less than 0 : cell %d   lifetime %d\n", i,ccl);
      }
      assert(ccl > 0);
      state.cellCycleLength[i] = (uint)ccl;
    }

    /*** Cell binary IDs ***/
    // char m[16];
    // int integ;
    // uint id_bit;

    // for(int i=0; i<state.numCells[0]; i++){

    //   integ = i;
    //   id_bit = 0;

    //   for(int g=5; g>=0; g--){
    //     if(integ-(pow(2,g))>=0){
    //       id_bit = id_bit | (1<<g);
    //       integ -= pow(2,g);
    //     }
    //   }

    //   state.cellId_bits[i] = id_bit;
    // }

    state.cellId_bits[0] = 0;

    /*** Cell Ligands ***/
    for(int i=0; i<state.numCells[0] * NUMLIGmax; i++){
      state.cellLigand[i] = 0.0;
    }

    /*** Cell protein ****/
    for(int i=0; i<state.numCells[0] * NUMPROTEINmax; i++){
      state.cellProtein[i] = 0.0;
    }    

    //save state as xml file
    save< State_Host >(state, "state_archive.xml");
  }
