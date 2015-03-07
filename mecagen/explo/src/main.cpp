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

// mg includes:
#include "metaparam.hpp"
#include "model.hpp"
#include "param_host.hpp"
#include "param_device.hpp"
// #include "param.hpp"
#include "state.hpp"
#include "thrust_objects.hpp"
#include "seaurchin_filelogger.hpp"
#include "controllerCreator.hpp"

// ISF include:
#include "controller.hpp"

// Standard
#include <iostream>
#include <stdlib.h>
#include <cstring>  //memset
#include <climits>  //memset
#include <time.h>  
#include <string>  
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


int main(int argc, char **argv)
{
 

  char adress[300];
  int repet = 0, ts = 0, te = 0;
  // double p1 = 0, p2 = 0;

  double wadh_homo = 100.0;
  double wadh_hete = 100.0;
  double wrep    = 100.0;
  double k_rig     = 9500.0;
  double coeff_gab = 1.0;
  double global_damping = 3000.0;
  double c_max     = 2.0; //1.2414;

  int outputfolder = 0;

  if(argc < 7){ 
    // std::cout << "Missing argument : ./mg_seaurchin_headless -ts 0 -te 3950 -outputfolder /path/to/folder -p1 100 -p2 100 -repet 10" << std::endl;
    std::cout << "Missing argument : ./mg_seaurchin_headless -ts 0 -te 4000 -openmole -p1 100 -p2 100 -repet 1" << std::endl;
    return 0;
  }

  for (int i = 1; i < argc; i++) {
    if(std::string(argv[i]) == "-outputfolder"){
      sprintf(adress, "%s/", argv[i+1] );
      outputfolder = 1;
    }
    else if(std::string(argv[i]) == "-openmole"){
      sprintf(adress, "%s", "./out/");
    }
    else if(std::string(argv[i]) == "-ts"){
      ts = atoi(argv[i+1]);
    }
    else if(std::string(argv[i]) == "-te"){
      te = atoi(argv[i+1]);
    }
    else if(std::string(argv[i]) == "-repet"){
      repet = atoi(argv[i+1]);
    }
    // else if(std::string(argv[i]) == "-p1"){
    //   p1 = atof(argv[i+1]);
    // }
    // else if(std::string(argv[i]) == "-p2"){
    //   p2 = atof(argv[i+1]);
    // }
    else if(std::string(argv[i]) == "-Waho"){
      std::istringstream istr(std::string(argv[i+1]));
      istr >> wadh_homo;  
    }
    else if(std::string(argv[i]) == "-Wahe"){
      std::istringstream istr(std::string(argv[i+1]));
      istr >> wadh_hete;
    }
    else if(std::string(argv[i]) == "-Wrep"){
      std::istringstream istr(std::string(argv[i+1]));
      istr >> wrep;
    }
    else if(std::string(argv[i]) == "-Krig"){
      std::istringstream istr(std::string(argv[i+1]));
      istr >> k_rig;
    }
    else if(std::string(argv[i]) == "-GabC"){
      std::istringstream istr(std::string(argv[i+1]));
      istr >> coeff_gab;
    }
    else if(std::string(argv[i]) == "-Damp"){
      std::istringstream istr(std::string(argv[i+1]));
      istr >> global_damping;
    }
    else if(std::string(argv[i]) == "-Cmax"){
      std::istringstream istr(std::string(argv[i+1]));
      istr >> c_max;
    }

  } 

  if(outputfolder == 1){
    sprintf(adress, "%s%.2lf_%.2lf_%.2lf_%.2lf_%.2lf_%.2lf_%.2lf/"
            ,adress, wadh_homo, wadh_hete, wrep, k_rig, coeff_gab, global_damping, c_max);
  }

  struct stat st = {0};

  if (stat(adress, &st) == -1) {
      mkdir(adress, 0700);
  }

  ////////////////////////////////////////
  ////////////////////////////////////////
  //////       Initialization     ////////
  ////////////////////////////////////////
  ////////////////////////////////////////
 
  /** Meta-Parameters **/
  mg::MetaParam<mg::HOST> meta;

  /** Parameters **/
  mg::Param_Host myParam;

  /** State **/
  mg::State_Host myState(&meta);

  /** Controller **/
  // isf::Controller_Impl_Host_Device<     
  //                   mg::MetaParam<mg::HOST>, 
  //                   mg::MetaParam<mg::DEVICE>, 
  //                   mg::Param<mg::HOST>, 
  //                   mg::Param<mg::DEVICE>, 
  //                   mg::State<mg::HOST>, 
  //                   mg::State<mg::DEVICE>, 
  //                   mg::Model
  //                                             > myController(meta);

  isf::Controller<
              mg::MetaParam<mg::HOST>, 
              mg::Param_Host, 
              mg::State_Host
                  >* myController = mg::createNewController(meta);
  
  
  ////////////////////////////////////////
  ////////////////////////////////////////
  //////     Repetition loop      ////////
  ////////////////////////////////////////
  ////////////////////////////////////////

  std::srand ( time(NULL) );

  for(int i=0; i<repet; i++){

    char adress_repet[400];
    sprintf(adress_repet, "%s/%d",adress, i);

    struct stat st = {0};

    if (stat(adress_repet, &st) == -1) {
        mkdir(adress_repet, 0700);
    }

    //generate random seeds
    int seedGauss = (int)(400000.0 * rand()/ (double) RAND_MAX);
    int seedUnif = (int)(400000.0 * rand()/ (double) RAND_MAX);

    char seeds_file[400];
    sprintf(seeds_file,"%s/seeds",adress_repet);
    std::fstream ofstream_seeds;
    ofstream_seeds.open(seeds_file, std::fstream::out);
    ofstream_seeds << seedGauss << " , " << seedUnif << std::endl;
    ofstream_seeds.close();

    myParam.initWithSeeds(seedGauss, seedUnif);

    // myParam.w_adh_homotypic[0]    = p1;
    // myParam.w_adh_heterotypic[0]  = p2;
    myParam.w_adh_homotypic[0]    = wadh_homo;
    myParam.w_adh_heterotypic[0]  = wadh_hete;
    myParam.w_rep[0]              = wrep;
    myParam.k_rig[0]              = k_rig;
    myParam.gab_coeff[0]          = coeff_gab;
    myParam.globalDamping[0]      = global_damping;
    myParam.c_max[0]              = c_max;

    myState.init(&myParam);

    /** Send the Param object to the Controller class. */
    myController->setParam(&myParam);

    // * Send the State object to the Controller class. 
    myController->setState(&myState);
    
    /** Specifies the initial and final time step of simulation. */
    // myController->setTimeInterval(myParam.initialTimeStep[0], //1000);
    //                       ((9-3) * 3600 + 35 * 60) / myParam.deltaTime[0]);    
    myController->setTimeInterval(ts,te);

    /** Consumers **/
    std::vector<isf::Consumer*> consumers;

    mg::SeaUrchinFileLogger filelogger(&meta, &myParam, adress_repet);
    consumers.push_back(&filelogger);

    // Add consumers to the controller
    myController->setConsumers(consumers);

    myController->start();

    int mainloop = true;

    while( myController->isOver() == false){
      usleep(500000);
    }

    filelogger.stop();
  }

  myController->stopAndJoin();

  // /** Command line interaction with the controller. **/
  // int mainloop = true;

  // while( myController->isOver() == false){
  //   usleep(500000);
  // }

  // myController->stopAndJoin();
  // //kill the Consumer threads
  // for(int i=0; i<(int)consumers.size(); i++){
  //  consumers[i]->stop();
  // }

  return 0;
}
