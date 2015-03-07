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

// Class definition:
#include "filelogger.hpp"

// Project:
#include "producerfactory.hpp"
#include "interfaces.hpp"
#include "tools.hpp"
#include "cells_producer.hpp"
#include "metaparam.hpp"
#include "param.hpp"
#include "define.hpp"

// Standard:
#include <iostream>
// #include <fstream>   //ifstream
#include <sstream>   //ostringstream
#include <cassert>
#include <unistd.h>
#include <cstdlib> //the standard C library header
#include <string>
#include <numeric>    //std::accumulate
#include <algorithm>  //std::transform

namespace mg{

  // Constructor
  FileLogger::FileLogger(MetaParam<HOST>* _mp, Param<HOST>* _p, char * _ofp, int _devtime_minutes_start, int _devtime_minutes_end, int _devtime_minutes_period)
  : 
    metaParam(_mp),
    param(_p),
    outputFolderPath(_ofp),
    devtime_minutes_start(_devtime_minutes_start),
    devtime_minutes_end(_devtime_minutes_end),
    devtime_minutes_period(_devtime_minutes_period)
  {
    // We specify the associated producers in the Consumer constructor, i.e. here. 
    // Each producer specification requires two arguments:
    //  - the producer ID, written "isf::get_id<***PRODUCERNAME***>()"
    //  - the period of the producer notification, written as an integer value
    isf::ProducerParameter pp = { isf::get_id<CellsProducer>(), 1 };
    producerParameters.push_back( pp ); 
    
    last_record = 0;
    files_closed = 0;
  }

  // Destructor
  FileLogger::~FileLogger() throw () { 
  }



  double mean(std::vector<double> *v){
    double sum = std::accumulate(v->begin(), v->end(), 0.0);
    return sum / v->size();
  }

  double stdev(std::vector<double> *v){
    double sum = std::accumulate(v->begin(), v->end(), 0.0);
    double mean = sum / v->size();

    std::vector<double> diff(v->size());
    std::transform(v->begin(), v->end(), diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    return std::sqrt(sq_sum / v->size());
  }

  // Child interface redefinition. 
  void FileLogger::consumerProcess(void * buffer, isf::ProducerFactory* prod_id, int step){

    // usleep(500000);

    //First of all, we detect which producer/broker as notified the consumer via the producer id
    if( prod_id == isf::get_id<CellsProducer>() ){
      
      Debug("\n FileLogger received a notification from broker " << (long)prod_id << " at Tstep " << step);

      //We cast the broker's buffer pointer with the export structure declared in the ForestProducer class.
      embryoState * buf = (embryoState *) buffer;
      
      //copie toutes les 3 minutes
      //3min = 180sec = 180 / SEAURCHINTS_SEC timesteps
      int devtime_minutes = (int)( ((double)(metaParam->devtime_minutes_init[0] + step * param->deltaTime[0])) / 60.0 );

      // std::cout << "step " << step << " devtime " << devtime_minutes << " lastrecord " << last_record
      //       << " diff " << (last_record != (uint)devtime_minutes) << " start " << devtime_minutes_start 
      //       << " end " << devtime_minutes_end << " period " << devtime_minutes_period << std::endl;

      if(
                 devtime_minutes >= devtime_minutes_start  
              && devtime_minutes < devtime_minutes_end
              && devtime_minutes%devtime_minutes_period == 0 
              && last_record != (uint)devtime_minutes
        ){ 

        last_record = devtime_minutes;

        //Create file storing neighbors
        char string[700];
        sprintf(string, "%s/grnstate_t%04d.csv", outputFolderPath, devtime_minutes);
        std::fstream ofstream_grnstate;
        // std::cout << "file " << string << std::endl;
        ofstream_grnstate.open(string, std::fstream::out);

        // std::cout << "record data in " << string << " path : " << outputFolderPath << std::endl;

        // Print first line
        ofstream_grnstate << "id";
        // std::cout << "id";
        for(uint i = 0; i < param->numProteins[0]; i++){
          ofstream_grnstate << ";" << param->proteins[i].name;
          // std::cout << ";" << param->proteins[i].name;
        }
        for(uint i = 0; i < param->numLigands[0]; i++){
          ofstream_grnstate << ";" << param->ligandParams[i].name;
          // std::cout << ";" << param->ligandParams[i].name;
        }
        ofstream_grnstate << std::endl;
        // std::cout << std::endl;

        // Print values
        for(uint id=0; id<buf->numCells; id++){
          ofstream_grnstate << id;
          // std::cout << "id " << id;
          for(uint i = 0; i < param->numProteins[0]; i++){
            ofstream_grnstate << ";" << buf->cellProtein[id*NUMPROTEINmax+i];
            // std::cout << ";" << param->proteins[i].name << ";" << buf->cellProtein[id*NUMPROTEINmax+i];
          }
          for(uint i = 0; i < param->numLigands[0]; i++){
            ofstream_grnstate << ";" << buf->cellLigand[id*NUMLIGmax+i];
            // std::cout << ";" << param->ligandParams[i].name << ";" << buf->cellLigand[id*NUMLIGmax+i];
          }
          ofstream_grnstate << std::endl;
          // std::cout << std::endl;
        }

        // Close stream
        ofstream_grnstate.close();

      }
      else if( devtime_minutes >= devtime_minutes_end && files_closed != 1){
        files_closed = 1;
        //close files

      }
      else{
        // printf("do not record data\n");
      }
    }
    else{
      std::cout << "\n FileLogger received a bad notification" << std::endl;
    }
  }
  
  void FileLogger::errorManagement(int step, int errorCode){

    std::cout << "FileLogger received an error notification (code " 
                  << errorCode << ") at time step " 
                  << step << std::endl;
    char string[300];
    sprintf(string, "%s/error", outputFolderPath);
    std::fstream ofstream_error;
    ofstream_error.open(string, std::fstream::out);
    ofstream_error << errorCode << std::endl;
    ofstream_error.close();
  }

};
