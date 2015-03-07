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

#ifndef _CONSUMER_H_2013_08
#define _CONSUMER_H_2013_08

// Project:
#include "eventmanager.hpp"   //pour le membre em
#include "interfaces.hpp"     //pour ConsumerConfig, ProducerParameter, NotifyParameter
#include "producerfactory.hpp"

// Standard :
#include "SDL_thread.h"

#include <vector>
// #include <thread>
#include <iostream>

namespace isf {

  /** The Consumer Class.
   * The Consumer is an abstract class from which the client-specified consumer classes inherits. It aims at dealing
   * with all the synchronization logic for the communication with the brokers in both
   * directions :
   *    - toward the consumer, the Event API is composed of two methods (notify(...), called by the broker after the swap, and
   * a stop() method to quit the consumer) and the EventManager deals with the thread synchronization.
   *    - from the consumer, the receiveNotification() notifies the broker after dealing with transfered data.
   */
  class Consumer {
    
    protected:
      /** Controller Constructor.
       *  The constructor is protected as it used by Controller_Impl but shall 
       *  remain hidden from the client code.
       */
      Consumer() ;

    public:
      /** Constroller Destructor. */
      virtual ~ Consumer() throw () ;

      // Event API:
      /** Push a ParametrizedEvent to the object's EventManager's deque. This public method is called by a broker
      * after it has swapped its buffers. The event is parametrized to identify the notifying broker. The event's 
      * action calls the internal private method receiveNotification()
      * @param[in] brid_      The notifying broker ID
      * @param[in] step_      The time step associated with the notification
      */
      void notify(ProducerFactory* brid_, int step_);

      /** Push a empty-parameter Parametrized to the object's EventManager's deque. This method is called when 
      * the consumer thread need to be stopped. The event's action calls the internal private method doStop()
      */
      void stop();

      // XXX
      void notifyError(int error_code, int step);
      
      // Public API:
      /** Set the class' ConsumerConfig object. This method is called by the controller after it has created the
      * producers and brokers in the setConsumers() method. 
      */
      void setConfig(ConsumerConfig cc);

      /** Returns the ProducerParameters specified by the client's consumer child class in the constructor method.
      * This method is called by the Controller in the setConsumers() method.
      */
      std::vector<ProducerParameter> getProducerParameters();

    protected:
      /** Pure virtual function implemented in the client-specified consumer child class. The child's method is the only 
      * method that the client has to implement. It contains the client consumer's action when it receives a notification. 
      * This method is called by the Consumer's receiveNotification() class.*/
      virtual void consumerProcess(void * buffer, ProducerFactory* prod_id, int step) = 0;
      
      //XXX
      virtual void errorManagement(int step, int errorCode) = 0;

      /** Is used to store the ConsumerConfig properties. */
      ConsumerConfig                        consumerConfig;

      /** Is used to store the list of producers and their period required by the client Consumer. Instanciated by the 
      * client in its own Consumer's constructor method.*/
      std::vector<isf::ProducerParameter>   producerParameters;

    private:
      // Internal data:
       /** Is used to store the events send by the other threads via the consumer public API. These events are
      * read and processed by the consumer thread. */
      EventManager<Consumer>*  em;

      /* Is used to store a pointer toward the controller execution thread. This pointer is instanciated in the 
      * constructor method and it calls the internal threadLoop() method. 
      */
      SDL_Thread *                executorThread;
      // C++11
      // std::thread*             executorThread;

      /** Is used to store the activity state of the thread. It is set to true in the class' constructor and only
      * changed to false by the internal doStop() method when the thread is asked to stop.
      */
      bool                     running;

      // Internal API:
      /** Main method of the Consumer class. It is called at construction and contains an loop controlled by the
      * running boolean variable. However, the loop is blocked in a passive state when no notification have been 
      * sent.
      */
      //static for SDL SDL_CreateThread
      static int threadLoop(void * ptr);
      
      /** Set the bool variable running to false.
      */
      void doStop(NotifyParameter p);

      /** Call the client-specified consumerProcess() when a broker notification event is read. Then, it notifies 
      * back the broker's eventmanagen.
      */
      void receiveNotification(NotifyParameter p);

      //XXX
      void receiveErrorNotification(NotifyParameter p);
  };


  
  /** This function simplifies the syntax used by the client to identify the producers. 
  *   As an example, " isf::get_id<ProducerX>() " 
  *   returns the producerFactory " isf::ProducerFactory_Impl<ProducerX>::getInstance() ", 
  *   which is the way producers are identified in ISF.
  */
  template<class P>
  ProducerFactory* get_id(){
    return ProducerFactory_Impl<P>::getInstance();
  }

} // End namespace

#endif
