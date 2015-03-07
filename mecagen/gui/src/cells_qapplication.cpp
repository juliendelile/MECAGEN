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

#include "cells_qapplication.hpp"

#include "serialization.hpp"

#include "controllerCreator.hpp"
#include "filelogger.hpp"

namespace mg{

  CellsQApplication::CellsQApplication(int & argc, char** argv):
    QApplication(argc, argv)
  {

    if(argc >= 7){

      // Create a vector of consumers and add the visualization consumer:
      std::vector<isf::Consumer*> consumers;

      // Default parameters:
      int tstart = 0, tend = 10000;
      // int seedGauss = -1;
      // int seedUnif = -1;
      std::string param_file;
      std::string metaparam_file;
      std::string state_file;
      uint filelogger_used = 0;
      sprintf(adress_filelogger, "/tmp");
      int fl_start = 0;
      int fl_end = NUMTIMESTEP;
      int fl_period = 1;

      // Read parameters:
      for (int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "-ts"){ tstart = atoi(argv[i+1]); }
        else if(std::string(argv[i]) == "-te"){ tend = atoi(argv[i+1]); }
        // else if(std::string(argv[i]) == "-seedGauss") { seedGauss = atoi(argv[i+1]); }
        // else if(std::string(argv[i]) == "-seedUnif")  { seedUnif  = atoi(argv[i+1]); }
        else if(std::string(argv[i]) == "-param")  { param_file  = std::string(argv[i+1]); }
        else if(std::string(argv[i]) == "-state")  { state_file  = std::string(argv[i+1]); }
        else if(std::string(argv[i]) == "-metaparam")  {  metaparam_file  = std::string(argv[i+1]); }
        else if(std::string(argv[i]) == "-flpath"){
          // Create the filelogger consumer and add it to the consumer's vector:
          sprintf(adress_filelogger, "%s", argv[i+1] );
          filelogger_used = 1;
        }
        else if(std::string(argv[i]) == "-flstart"){
          fl_start = atoi(argv[i+1]);
          filelogger_used = 1;
        }
        else if(std::string(argv[i]) == "-flend"){
          fl_end = atoi(argv[i+1]);
          filelogger_used = 1;
        }
        else if(std::string(argv[i]) == "-flperiod"){
          fl_period = atoi(argv[i+1]);
          filelogger_used = 1;
        }
      }      
      
      bool filesexist = true;
      
      // Load metaparam
      metaParam = new MetaParam<HOST>();
      if (FILE *file = fopen(metaparam_file.c_str(), "r")) {
          fclose(file);
          load< MetaParam<HOST> >(*metaParam, metaparam_file.c_str());
      } else {
         printf("Could not find metaparam file.\n");
         filesexist = false;
      }

      // Load param
      param = new Param_Host();
      if (FILE *file = fopen(param_file.c_str(), "r")) {
          fclose(file);
          load< Param_Host >(*param, param_file.c_str());
      } else {
         printf("Could not find param file.\n");
         filesexist = false;
      }

      // Load state
      state = new State_Host(metaParam);
      if (FILE *file = fopen(state_file.c_str(), "r")) {
          fclose(file);
          load< State_Host >(*state, state_file.c_str());
      } else {
         printf("Could not find state file.\n");
         filesexist = false;
      }

      if( filesexist == true ){
        
        window = new Window(&controller, metaParam, param, state, this);

        vizconsumer = new VizConsumer(window->glWidget);
        consumers.push_back(vizconsumer);

        if(filelogger_used == 1){
          filelogger = new FileLogger(metaParam, param, adress_filelogger, fl_start, fl_end, fl_period);
          consumers.push_back(filelogger);
        }

        // Create the controller
        controller = mg::createNewController(*metaParam);

        controller->setParam(param);
        controller->setState(state);
        // controller->setTimeInterval(statec.urrentTimeStep[0], state.currentTimeStep[0] + 5000);
        controller->setConsumers(consumers); // Add consumers to the controller
        
        // Initialize the window:
        window->initGUI(tstart, tend);
        window->resize(window->sizeHint());
        // window.setAttribute( Qt::WA_QuitOnClose, true );
        window->show();

        // int desktopArea = QApplication::desktop()->width() * QApplication::desktop()->height();
        // int widgetArea  = window.width() * window.height();
        // if (((float)widgetArea / (float)desktopArea) < 0.75f) {window.show();} else {window.showMaximized();}

        // to notify the visconsumer, and thus allow displaying before starting the simulation:
        // controller->executeProducers();


        // enable producer for display
        controller->setTimeInterval(0, 0);
        controller->start();  

        init_ok =true;
      }
      else{
        init_ok = false;
      }
    }
    else{ //wrong arguments
      std::cout << "\n\n*** Missing argument : ./launchGUI.sh -param /path/to/paramfile.xml -metaparam /path/to/metaparam.xml -state /path/to/state.xml " << std::endl;
      init_ok = false;
    }
  }

  // Destructor
  CellsQApplication::~CellsQApplication() throw () {
    controller->stopAndJoin();
    delete vizconsumer;
    delete window;
    delete controller;
  }
}
