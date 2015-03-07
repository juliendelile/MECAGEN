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
    
    
    // load and save FlatTissueFrom1cell_grn_default metaParam archive (equivalent to copy...)
    MetaParam<HOST>     metaParam;
    load< MetaParam<HOST> >(metaParam, "../../FlatTissueFrom1cell_grn_default/output880cells/metaparam_archive.xml");
    save< MetaParam<HOST> >(metaParam, "metaparam_archive.xml");
    
    // load FlatTissueFrom1cell_grn_default state archive
    State_Host state(&metaParam);
    load< State_Host >(state, "../../FlatTissueFrom1cell_grn_default/output880cells/state_archive.xml");
    
    /*** Cell Ligands ***/
    // for(int i=0; i<state.numCells[0] * NUMLIGmax; i++){
    //   state.cellLigand[i] = 156.0;
    // }

    // // Specific to Signal mediated toggle switch : Receptor_1, Receptor_2, STF and Silencer proteins are already present in the cell 

    
    for(int i=0; i<state.numCells[0]; i++){

      /*** Cell protein ****/
      for(int j=0; j < NUMPROTEINmax; j++){
        if(j==1 || j==2 || j==5 || j==6){
          state.cellProtein[i*NUMPROTEINmax+j] = 60.0;
        }
        else if(j == 8){    // Inhibitor_TF
          state.cellProtein[i*NUMPROTEINmax+j] = 43.2;  
        }
        else{
          state.cellProtein[i*NUMPROTEINmax+j] = 0.0;
        }
      }

      // state.cellCycleLength[i] = 100000;
    }    



    //save state as xml file
    save< State_Host >(state, "state_archive.xml");
  }
