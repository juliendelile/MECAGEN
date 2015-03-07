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

#ifndef _FileLogger_HPP_
#define _FileLogger_HPP_

#include "thrust_objects.hpp"     // Backend<>

// ISF Class:

#include "consumer.hpp"

#include <fstream>   //ifstream

// Standard:
// #include <vector>

namespace isf {
    class ProducerFactory;
  }

namespace mg {

  //Forwand declaration
  template<int T> class MetaParam;
  template<int T> class Param;

  /** FileLogger, example of client-specified consumer class. 
  * This consumer works in association with two producers, forest_producer and forest_producer_counter.
  */
  class FileLogger : public isf::Consumer {

      public:
        /** FileLogger class constructor. 
        * This constructor takes as argument a pointer toward a host MetaParam object, this is for the sake 
        * of examplarity. Other consumer may not need such argument.
        */
        FileLogger(MetaParam<HOST>* _mp, Param<HOST>* _p, char * _ofp, int _devtime_minutes_start, int _devtime_minutes_end, int _devtime_minutes_period);

        /** FileLogger class destructor. */
        virtual ~FileLogger() throw ();

      protected:
        
        /** Implements the action of the consumer when it receives a notification. 
        * @param[in] buffer   Pointer toward the notifying broker buffer
        * @param[in] prod_id  Id of the notifying broker
        * @param[in] step     Timestep of the notification
        */
        void consumerProcess(void * buffer, isf::ProducerFactory* prod_id, int step);

        //XXX
        void errorManagement(int step, int errorCode);

      private: 
        /** Is used to store a pointer toward a host MetaParam object. It may be not necessary to implement
        * it in other user context. */
        MetaParam<HOST>* metaParam;
        Param<HOST>* param;
        char * outputFolderPath;

        uint last_record;
        uint files_closed;
        int devtime_minutes_start;
        int devtime_minutes_end;
        int devtime_minutes_period;

        // double  fitness_linkdistance;
        // int     fitness_linkdistance_counter;
        // double  fitness_sphericity;
        // int     fitness_sphericity_counter;
        // double  fitness_normale;
        // int     fitness_normale_counter;
        // double  fitness_planarity;
        // int     fitness_planarity_counter;

        // std::fstream ofstream_lineage;
        // std::fstream ofstream_neighbors;
    };
};
#endif
