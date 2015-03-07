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

// Class definition:
#include "broker.hpp"

// Project
#include "eventmanager.hpp"
#include "consumer.hpp"  // Only needed for notification !
#include "tools.hpp" 

// Dependencies:
#include <cstdlib>
#include <cassert>
#include <iostream>



namespace isf {

  // Constructor
  Broker::Broker(BrokerConfig conf, size_t allocationSize, ProducerFactory* brid) :
    brokerConfig(conf),notificationCounter(0),em(new EventManager<Broker>()), brokerID(brid)
  {
    doAllocation(allocationSize);
  }

  // Destructor
  Broker::~Broker() throw () {
    delete em;
    free(producerBuffer);
    free(consumerBuffer);
  }

  // Event API:
  void Broker::releaseBuffer(){

    Debug("Broker.releaseBuffer: broker " << (long)this << " is notified by a consumer");

    em->send(new Event<Broker>(this, &Broker::doReleaseBuffer));
  }

  // Public API:
  void Broker::swapBuffers(int step){

    // Wait for notifications:
    while(notificationCounter > 0){
      Event<Broker>* evt = em -> read();
      evt->process();
      delete evt;
    }

    // Swap:
    void * tmp = producerBuffer;
    producerBuffer = consumerBuffer;
    consumerBuffer = tmp;

    // Notify consumers:
    notifyConsumers(step);
  }

  void * Broker::getProducerBuffer(){
    return producerBuffer;
  }

  void * Broker::getConsumerBuffer(){
    return consumerBuffer;
  }

  // // Internal functions:
  void Broker::doReleaseBuffer(){
    --notificationCounter;
  }

  void Broker::doAllocation(size_t size){
    producerBuffer = malloc(size);
    consumerBuffer = malloc(size);
  }

  void Broker::notifyConsumers(int step){

    assert(notificationCounter==0);

    for(std::map<Consumer*, std::set<int> >::iterator it = brokerConfig.config.begin(); 
            it != brokerConfig.config.end(); ++it){

      bool doNotify = false;
      
      // for(auto itSet = it->second.cbegin(); !doNotify && itSet != it->second.cend(); ++itSet){
      
      for(std::set<int>::iterator itSet = it->second.begin(); 
              !doNotify && itSet != it->second.end(); ++itSet){
       
        doNotify = (step % *itSet) == 0;
      }
      if(doNotify){
        Debug("Broker.notifyConsumer: broker " << (long)this 
            << " notifies consumer " 
            << (long)(it->first));
        
        //notify associated consumer
        it->first->notify(brokerID, step); 
        // std::cout << "sleep 1 " << std::endl;
        // std::this_thread::sleep_for( std::chrono::seconds(1));
        ++notificationCounter;
        // std::cout << "sleep 2 " << std::endl;
        // std::this_thread::sleep_for( std::chrono::seconds(1));
      }
    }
  }

} // End namespace
