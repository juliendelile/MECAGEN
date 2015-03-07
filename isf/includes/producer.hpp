 /*
 * Copyright (C) 2015 by Julien Delile and Matthieu Herrmann
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

#ifndef _OUTPUT_H_2013_08
#define _OUTPUT_H_2013_08

// Project definitions
#include "interfaces.hpp"
#include "broker.hpp"
#include "tools.hpp"

// Standard C definition
#include <iostream>



namespace isf {

  // Top level interface
  /** The Producer Class.
   * The Producer class is a templated class that is the top level of the producer concept. It is the parent class
   * of the ISF-provided Producer_Impl_Host_Only and Producer_Impl_Host_Device classes and the grand-parent of the
   * client-provided producer class. It implements the mechanism of execution of the producers and the control of the
   * broker swapping.
   */
  template<typename MetaParam, typename State, typename StateDevice>
    class Producer{
      public:

        /** Producer constructor. */
        Producer(){};

        /** Producer destructor. */
        virtual ~Producer() throw (){};

        //Public API

        /** Specifies the ProducerConfig object and the client-provided MetaParam object that will be used in the
         * mustProcess() and execute() methods. This method is called by the controller's setConsumer() method. */
        void setConfig(ProducerConfig pc, MetaParam * mp);

        /** Specifies the associated broker pointer which will be called to access the broker's buffered data and
         * the broker's swap method.
         */
        void setBroker(Broker * br);

        /** Calls the doProcessing() method implemented by the client. It first checks if the time step
         * corresponds to a execution time step and, after processing the data, it asks the associated broker
         * to swap its buffers.
         */
        void execute(State* s, StateDevice* sd, int step, int initialTS);

        /** Checks if the producer must execute at the timestep step.*/
        bool mustProcess(int step);

        /** Pure virtual function implemented in the client-specified producer child class. This method retuns the
         * size required by the produced object. It is use to allocate the broker buffer's memory.
         */
        virtual size_t getRequestedBufferSize() = 0;

      protected:
        /** Pure virtual function implemented in the client-specified producer child class. This method implements
         * the processing performs by the producer and the actual writing into the broker's buffer.
         * @param[in] s          The host state from which the data are produced
         * @param[in] sd         The device state from which the data are produced
         * @param[in] step       The time step of production
         * @param[out] buffer    The pointer toward the broker's buffer
         */
        virtual void calldoProcessing(State* s, StateDevice* sd, int step, void * buffer) = 0;

        /** Is used to store the pointer toward the associated broker. */
        Broker *        broker;

        /** Is used to store the pointer toward the MetaParam specified by the client. */
        MetaParam *     metaParam;

      private:
        /** Is used to store the ProducerConfig object.*/
        ProducerConfig      producerConfig;

    };

  /** Producer_Impl_Host_Device class.
   * This class inherits from the Producer class and it exists for compilation sake. Indeed, the Controller class
   * must store a vector of Producer but do not know the client-specified PChild. This class is the parent class
   * of the client-provided producer classes. It also serves as a proxy between its parent class method "execute" which
   * calls the "calldoProcessing" method implemented here, and the client-specified method "doProcessing" which is
   * declared here with the adapted argument signature: "State* s, StateDevice* sd, int step, void * buffer".
   */
  template<typename MetaParam,  typename State, typename StateDevice>
    class Producer_Impl_Host_Device: public Producer<MetaParam, State, StateDevice>{
      public:
        /** Producer_Impl_Host_Device constructor. */
        Producer_Impl_Host_Device(){};

        /** Producer_Impl_Host_Device destructor. */
        virtual ~Producer_Impl_Host_Device() throw (){};

        /** Child Implementation of Producer's "calldoProcessing" method which serves as a proxy for the client-specified method
         * "doProcessing".
         */
        void calldoProcessing(State* s, StateDevice* sd, int step, void * buffer);

        /** Pure virtual function implemented in the client-specified producer child class. */
        virtual void doProcessing(State* s, StateDevice* sd, int step, void * buffer) = 0;

      private:

    };


  /** Producer_Impl_Host_Only class.
   * This class inherits from the Producer class and it exists for compilation sake. Indeed, the Controller class
   * must store a vector of Producer but do not know the client-specified PChild. This class is the parent class
   * of the client-provided producer classes. It also serves as a proxy between its parent class method "execute" which
   * calls the "calldoProcessing" method implemented here, and the client-specified method "doProcessing" which is
   * declared here with the adapted argument signature: "State* s, int step, void * buffer".
   */
  template<typename MetaParam, typename State>
    class Producer_Impl_Host_Only: public Producer<MetaParam, State, State>{
      public:
        /** Producer_Impl_Host_Only constructor. */
        Producer_Impl_Host_Only(){};

        /** Producer_Impl_Host_Only destructor. */
        virtual ~Producer_Impl_Host_Only() throw (){};

        /** Child Implementation of Producer's "calldoProcessing" method which serves as a proxy for the client-specified method
         * "doProcessing".
         */
        void calldoProcessing(State* s, State* sd, int step, void * buffer);

        /** Pure virtual function implemented in the client-specified producer child class. */
        virtual void doProcessing(State* s, int step, void * buffer) = 0;

      private:

    };

  //
  // Definition Producer<MetaParam, State>:
  //

  template <typename MetaParam, typename State, typename StateDevice>
    void Producer<MetaParam,State,StateDevice>::setConfig(ProducerConfig pc, MetaParam * mp){
      producerConfig = pc;
      metaParam = mp;
    }

  template <typename MetaParam, typename State, typename StateDevice>
    void Producer<MetaParam,State,StateDevice>::setBroker(Broker * br){
      broker = br;
    }

  template <typename MetaParam, typename State, typename StateDevice>
    void Producer<MetaParam,State,StateDevice>::execute(State* s, StateDevice* sd, int step, int initialTS){
      // Frequences management:
      Debug("\nProducer.execute: producer : " << (long)this << " mustProcess at t = " << step << " -> " <<
          mustProcess(step-initialTS) );

      // In the classical usage of the producer, the periods of producer execution are expressed from the initial time step.
      // Custom mustProcess methods may be specified by the client using the XXX child ProducerXXX
      if(mustProcess(step-initialTS)){
        // Execution of post-processing in free buffer:
        void * buffer = broker->getProducerBuffer();

        calldoProcessing(s, sd, step, buffer);

        broker->swapBuffers(step);
      }
    }

  template <typename MetaParam, typename State, typename StateDevice>
    void Producer_Impl_Host_Device<MetaParam,State,StateDevice>::calldoProcessing(State* s, StateDevice* sd, int step, void* buffer){

      doProcessing(s, sd, step, buffer);
    }

  template <typename MetaParam, typename State>
    void Producer_Impl_Host_Only<MetaParam, State>::calldoProcessing(State* s, State* sd, int step, void * buffer){

      doProcessing(s, step, buffer);
    }

  template <typename MetaParam, typename State, typename StateDevice>
    bool Producer<MetaParam,State,StateDevice>::mustProcess(int step){

      bool ret = false;

      for( std::set<int>::iterator it = producerConfig.config.begin();
          !ret && it != producerConfig.config.end(); ++it ){
        ret = (step % *it) == 0;
      }

      // C++11 version
      // for( auto it = producerConfig.config.cbegin(); !ret && it != producerConfig.config.cend(); ++it ){ ... }

      return ret;
    }

} // End namespace
#endif
