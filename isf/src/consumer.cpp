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
#include "consumer.hpp"

// Project:
#include "broker.hpp"
#include "tools.hpp"

#include "SDL.h"
#include "SDL_thread.h"

// Standard:
#include <iostream>



namespace isf {

  // Constructor
  Consumer::Consumer() 
  :
    em(new EventManager<Consumer>()),
    running(true) 
    {
      executorThread = SDL_CreateThread(threadLoop, "threadLoop", this);
      //C++11 
      // executorThread = new std::thread(&Consumer::threadLoop, this);
    }

  // Destructor
  Consumer::~Consumer() throw () { 
    //C++11
    // delete executorThread; 

    delete em; 
  }

  // Event API:
  void Consumer::notify(ProducerFactory* brid_, int step_){
    
  Debug("Consumer.notify: consumer " << (long)this << " is notified by " << (long)consumerConfig.config[brid_] 
        << " at t = " << step_);

    NotifyParameter param;
    param.brokerID = brid_;
    param.step  = step_;

    em->send(new ParametrizedEvent<Consumer, NotifyParameter>(this, &Consumer::receiveNotification, param));
  }

  void Consumer::stop(){
    NotifyParameter param;  //useless param in this case
    param.brokerID = NULL;
    param.step  = 0;
    em->send(new ParametrizedEvent<Consumer, NotifyParameter>(this, &Consumer::doStop, param));
    int status;
    SDL_WaitThread(executorThread, &status);

    //C++11
    // executorThread->join();
  }

  void Consumer::notifyError(int error_code, int step){
    NotifyParameter param; 
    param.brokerID = NULL; //unused here
    param.step  = step;
    param.errorCode = error_code;
    em->send(new ParametrizedEvent<Consumer, NotifyParameter>(this, &Consumer::receiveErrorNotification, param));
  }

  void Consumer::setConfig(ConsumerConfig cc){
    consumerConfig = cc;
  }

  std::vector<isf::ProducerParameter> Consumer::getProducerParameters(){
    return producerParameters;
  }

  // Internal API:
  void Consumer::doStop(NotifyParameter p){ running = false; }

  //http://www.gamedev.net/topic/274615-passing-a-member-function-to-sdl_createthread/ for void * ptr
  int Consumer::threadLoop(void * ptr){
    // std::cout << "Consumer.threadLoop: start " << std::endl;
    while(((Consumer*)ptr)->running){
      // std::cout << "Consumer.threadLoop: new iteration " << std::endl;
      //ParametrizedEvent<Consumer, NotifyParameter> evt = em -> read();
      Event<Consumer>* evt = ((Consumer*)ptr)->em -> read();
      evt->process();
      delete evt;
      // std::cout << "Consumer.threadLoop: process over " << std::endl;
    }

    return 0;
  }

  void Consumer::receiveNotification(NotifyParameter p){
    
    void * buffer = consumerConfig.config[p.brokerID]->getConsumerBuffer();

    consumerProcess(buffer, p.brokerID, p.step); //implemented in client class

    //notifies the broker that data have been processed
    Debug("Consumer.receiveNotif: consumer " << (long)this 
            << " notifies broker " 
            << (long)(consumerConfig.config[p.brokerID]));

    consumerConfig.config[p.brokerID]->releaseBuffer();
  }

  void Consumer::receiveErrorNotification(NotifyParameter p){
    errorManagement(p.step, p.errorCode); //implemented in client class
  }

} // End namespace
