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
#include "cells_visualizer.hpp"

// Standard includes:
#include <iostream>
#include <cassert>
#include <unistd.h>

namespace mg{

  // Constructor
  VizConsumer::VizConsumer(GLWidget *_glw):
    glWidget(_glw)
  {
    // We specify the associated producers in the Consumer constructor, i.e. here.
    // Each producer specification requires two arguments:
    //  - the producer ID, written "isf::get_id<***PRODUCERNAME***>()"
    //  - the period of the producer notification, written as an integer value
    isf::ProducerParameter pp = { isf::get_id<CellsProducer>(), 1 };
    producerParameters.push_back( pp );   // Field of isf::Consumer
  }
  
  // Destructor
  VizConsumer::~VizConsumer() throw () { }
  
  // Child interface redefinition.
  void VizConsumer::consumerProcess(void * buffer, isf::ProducerFactory* prod_id, int step){
    //First of all, we detect which producer/broker as notified the consumer via the producer id
    if( prod_id == isf::get_id<CellsProducer>() ){
      Debug("\n VizConsumer received a notification from broker " << (long)prod_id << " at Tstep " << step);
      // We cast the broker's buffer pointer with the export structure declared in the ForestProducer class.
      embryoState * buf = (embryoState *) buffer;
      // Swap the display buffer
      glWidget->setBuffer(buf, step);
      // Sleep for a while: thie will eventually block the simulation as the notification from this consumer
      // is only send at the end of this method.
      // usleep(1000);
    }
    else{ std::cerr << "\n VizConsumer received a bad notification" << std::endl; }
  }

};
