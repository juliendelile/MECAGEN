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

#ifndef _CONTROLLER_HPP_2013_08
#define _CONTROLLER_HPP_2013_08

//----------------------------------------------------------------------------------------------------------------------
// Include Project:
#include "consumer.hpp"
#include "producer.hpp"
#include "producerfactory.hpp"
#include "broker.hpp"
#include "tools.hpp"
#include "modelproxies.hpp"

#include "SDL_customized_objects.h"

#include "SDL.h"
#include "SDL_atomic.h"
#include "SDL_thread.h"

// Standard:
#include <vector>
#include <cassert>
#include <map>
#include <unistd.h>
#include <sys/time.h>
// #include <iostream>

// C++11 version
// #include <atomic>
// #include <condition_variable>
// #include <thread>
// #include <mutex>
// #include <chrono>
//----------------------------------------------------------------------------------------------------------------------

namespace isf {

  /** The Controller Class.
   * The controller managed the execution of a model.
   * Controller is an abstraction of the controller used in the framework while
   * Controller_Impl is the real implementation.
   */
  template<typename MetaParam, typename Param, typename State>
    class Controller{

      protected:
        /** Controller Constructor.
         *  The constructor is protected as it used by Controller_Impl but shall
         *  remain hidden from the client code.
         */
        Controller();

      public:
        /** Constroller Destructor. */
        virtual ~Controller() throw ();

        //
        // Public API:
        //

        /** Pure virtual function implemented in the Controller_Impl class. */
        virtual void setState(State* p) = 0;

        /** Pure virtual function implemented in the Controller_Impl class. */
        virtual void setParam(Param* p) = 0;
              
        // virtual Param getParam() = 0;

        virtual State getState() = 0;

        // Step API:

        /** Specify the initial and final time step of the current simulation loop. */
        void setTimeInterval(int timeBegin, int timeEnd);

        /** Pure virtual function implemented in the Controller_Impl class. */
        virtual void setConsumers(std::vector<Consumer*> consumers) = 0;

        // Event API:

        /** Starts the simulation loop with the latest available data model (State and Param objects)
         * and reinitializes the time step. The simulation thread enters the simulation loop.
         */
        int  start();

        /** Stops the simulation loop and waits for the simulation thread to finish.  */
        void stopAndJoin();

        /** Pause or unpause the simulation loop. Useful only if interactive visualization
         * is running. */
        void playPause();

        // Interrogation API:

        /** Returns true if the simulation loop is paused. */
        bool isPause();

        /** Returns true if the final time step has been reached. */
        bool isOver();

        /** Returns true if the final time step has been reached. */
        int getCurrentTimeStep();

      protected:

        //XXX: public here as gui application may need to call the function to allow display before
        // simulation is started
        virtual void executeProducers() = 0;

        // WARNING ! ORDER MATTERS !
        // Internal Data:

        /** Is used to protect the Param buffer during modification. */
        Mutex  paramLock;
        // C++11 version
        // std::mutex  paramLock;

        /** Is used to protect the State buffer during modification. */
        Mutex  stateLock;
        // C++11 version
        // std::mutex  stateLock;

        /** Is used to set (and unset) the simulation loop into pause. */
        SDL_atomic_t   modelPause;
        // std::atomic_bool modelPause;

        /** Is used to exit the simulation loop when the simulation thread is finished
         * by setting it to false. */
        SDL_atomic_t  modelRunning;
        // C++11 version
        // std::atomic_bool modelRunning;

        /** Is used to record the following test: true if the last simulation step has been reached,
         * false otherwise. */
        SDL_atomic_t modelOver;
        // C++11 version
        // std::atomic_bool modelOver;

        /** Is used to record if the Param data has been updated. */
        SDL_atomic_t paramUpdated;
        // C++11 version
        // std::atomic_bool paramUpdated;

        /** Is used to record if the State data has been updated. */
        SDL_atomic_t stateUpdated;
        // C++11 version
        // std::atomic_bool stateUpdated;

        // Condition variable:

        /** This condition variable deals with the synchronization between the simulation thread
         *  and the controller API.
         */
        ConditionVariable   modelExecCV;
        // C++11 version
        // std::condition_variable_any   modelExecCV;

        /** Is required to implement the modelExecCV condition variable. */
        Mutex              modelExecLock;
        // C++11 version
        // std::mutex         modelExecLock;

        /** Is required to unlock the modelExecCV condition variable when the controller API is used. */
        bool               modelExecAccessLoop;

        // Public API Data:

        /** Is used to store the initial time step. Are atomic_int so the threads do not have
         *   to be locked during modification. */
        SDL_atomic_t modelTBegin;
        // C++11 version
        // std::atomic_int modelTBegin;

        /** Is used to store the final time step. Are atomic_int so the threads do not have
         *   to be locked during modification. */
        SDL_atomic_t modelTEnd;
        // C++11 version
        // std::atomic_int modelTEnd;

        /** Is used to store the current simulation time step. */
        int modelT;

        // Polymorphic internal API:



        /** Pure virtual function implemented in the Controller_Impl class. */
        virtual void swapStatePointer() = 0;

        /** Pure virtual function implemented in the Controller_Impl class. */
        virtual void swapParamPointer() = 0;

        /** Pure virtual function implemented in the Controller_Impl_Host_Only and Controller_Impl_Host_Device classes. */
        virtual void loadDeviceState() = 0;

        /** Pure virtual function implemented in the Controller_Impl_Host_Only and Controller_Impl_Host_Device classes. */
        virtual void loadDeviceParam() = 0;

        /** The internal ModelExecutor class.
         *  Encapsulated the excecution thread runnig the simulation.
         */
        class ModelExecutor{
          public:

            /** ModelExecutor Constructor. */
            ModelExecutor(){};

            /** ModelExecutor Destructor. */
            virtual ~ModelExecutor() throw (){};

            // API:

            /** Pure virtual function implemented in the ModelExecutor_Impl class defined in the
             *  Controller_Impl class. */
            virtual void join() = 0;
        };

        /** Is used to reference the ModelExecutor object. Must be provided by the ModelExecutor_Impl class. */
        ModelExecutor *  modelExecutor;
    };


  /** The Controller_Impl Class.
   * The implementation of the controller by inheritance. It is instanciated by the client code.
   * The class is templated in order to:
   *    - allow the creation of templated Producers objects,
   *    - deal with client-provided classes.
   * Requirements: the MetaParam, Param and State instances are allocated in the Controller_Impl
   * so these objects must follow the Rule of the Great Three (copy assignment operator is used here). Most methods of
   * the controller concept are implemented in this class, however, the "loadDeviceState" and "loadDeviceParam"
   * methods are implemented in the Controller_Impl_Host_Only and Controller_Impl_Host_Device derived class.
   */
  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
          class Controller_Impl: public Controller<MetaParam, Param, State> {

            public:
              /** Controller_Impl Constructor. MetaParam is passed as an argument to allow State
               * objects initializaion. The modelExecutor pointer from the parent Controller class
               * also initialized here.
               */
              Controller_Impl(MetaParam mp_);
              // Note: in a future release, the controller may take framework parameters such as GPGPU
              // config.
              //Controller(ISFParam isfp_, MetaParam mp_);

              /** Controller_Impl Destructor. */
              virtual ~Controller_Impl() throw ();

              // Configuration API:


              /** Create the producers and brokers objects, depending of the client-specified
               * consumer objects
               */
              void setConsumers(std::vector<Consumer*> cons);

              /** Execute all the producers. Is called during the simulation loop.*/
              void executeProducers();

              //XXX
              void sendErrorToConsumers(int error_code, int step);

            protected:



              /** Copy the Param object pointed by the argument p in the Controller_Impl's instance
               * of the Param object (ctlParamPublic). The method is defined in Controller_Impl as it
               * requires the Param template knowledge.
               */
              void setParam(Param* p);

              /** Copy the State object pointed by the argument p in the Controller_Impl's instance
               * of the State object (ctlStatePublic). The method is defined in Controller_Impl as it
               * requires the State template knowledge.
               */
              void setState(State* s);

              // Param getParam();

              virtual State getState() = 0;

              /** Pure virtual function implemented in the Controller_Impl_Host_Only and Controller_Impl_Host_Device classes. */
              virtual void loadDeviceState() = 0;

              /** Pure virtual function implemented in the Controller_Impl_Host_Only and Controller_Impl_Host_Device classes. */
              virtual void loadDeviceParam() = 0;


              // WARNING ! ORDER MATTERS !
              // The order of construction is the order of declaration !
              // ctlMetaParam is referenced by ctlState1 and ctlState2.
              // Hence, it must be constucted before ctlState1 and ctlState2.

              /** Is used to store the MetaParam instance of Controller_Impl. */
              MetaParam   ctlMetaParam;

              /** Is used to store one of the two Param instances of Controller_Impl. */
              Param       ctlParam1;

              /** Is used to store one of the two Param instances of Controller_Impl. */
              Param       ctlParam2;

              /** Is used to store the pointer toward the externally accessible Param buffer.*/
              Param *     ctlParamPublic;

              /** Is used to store the pointer toward the internally accessible Param buffer.*/
              Param *     ctlParamIntern;

              /** Is used to store one of the two State instances of Controller_Impl. */
              State       ctlState1;

              /** Is used to store one of the two State instances of Controller_Impl. */
              State       ctlState2;

              /** Is used to store the pointer toward the externally accessible State buffer.*/
              State *     ctlStatePublic;

              /** Is used to store the pointer toward the internally accessible State buffer.*/
              State *     ctlStateIntern;

              /** Is used to store the ParamDevice on the device backend */
              ParamDevice          ctlParamDevice;

              /** Is used to store the State on the device backend */
              StateDevice          ctlStateDevice;

              /** Is used to store all the producers created by the framework. */
              std::vector<Producer<MetaParam, State, StateDevice>*>  producers;

              /** Is used to store all the brokers created by the framework. */
              std::vector<Broker*>                      brokers;

              //XXX
              std::vector<Consumer*> consumers;




              // Internal functions:

              /* Swapping mechanism: the ctlParamInterm is transfered to the ctlParamPublic and
               * vice-versa.
               */
              void swapStatePointer();

              /* Swapping mechanism: the ctlStateInterm is transfered to the ctlStatePublic and
               * vice-versa.
               */
              void swapParamPointer();

            public:

              // virtual void callAlgoStep1(Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>* controller);
              // virtual void callAlgoStep2(Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>* controller);

              /** ModelExecutor_Impl class.
               *  Encapsulated the execution thread running the simulation.
               */
              class ModelExecutor_Impl: public Controller<MetaParam, Param, State>::ModelExecutor {
                public:

                  /** ModelExecutor_Impl Constructor. */
                  ModelExecutor_Impl(Controller_Impl* ci);

                  /** ModelExecutor_Impl Destructor. */
                  virtual ~ModelExecutor_Impl() throw ();

                  // API:
                  /** Blocks the main thread until the simulation executorThread thread finishes
                   * its execution.
                   */
                  virtual void join();

                  // Operational function:
                  /* Contains the simulation loop and its controls. It is started as an independent thread.
                  */
                  static int threadLoop(void * ptr);

                  // Data:
                  /* Is used to store a pointer toward the owner class Comtroller_Impl. */
                  Controller_Impl* controller_i;

                  /* Is used to store a pointer toward the simulation execution thread. */
                  SDL_Thread* executorThread;
                  // std::thread* executorThread;
              };

              //http://stackoverflow.com/questions/605497/accessing-inherited-variable-from-templated-parent-class
              using Controller<MetaParam, Param, State>::modelExecutor;
              using Controller<MetaParam, Param, State>::paramLock;
              using Controller<MetaParam, Param, State>::paramUpdated;
              using Controller<MetaParam, Param, State>::stateLock;
              using Controller<MetaParam, Param, State>::stateUpdated;
              using Controller<MetaParam, Param, State>::modelT;
              using Controller<MetaParam, Param, State>::modelTBegin;
          };

  /** The Controller_Impl_Host_Only Class.
   * This class inherits from the Controller_Impl and it aims at implementing the "loadDeviceState" and "loadDeviceParam"
   * methods which are called, but do not operate in the Host-Only mode of the framework. This class is the one implemented
   * in the client code.
   */
  template<typename MetaParam, typename Param, typename State, typename Model>
    class Controller_Impl_Host_Only
    : public Controller_Impl<MetaParam, Param, Param, State, State, Model> {

      public:

        /** Controller_Impl_Host_Only Constructor. This constructor is used in the client code.*/
        Controller_Impl_Host_Only(MetaParam mp_) :
          Controller_Impl<MetaParam, Param, Param, State, State, Model>(mp_){}

        /** Empty method in the Host-Only mode.*/
        void loadDeviceState();

        /** Empty method in the Host-Only mode.*/
        void loadDeviceParam();

        State getState();

        using Controller_Impl<MetaParam, Param, Param, State, State, Model>::stateLock;
        using Controller_Impl<MetaParam, Param, Param, State, State, Model>::ctlStateIntern;
    };

  /** The Controller_Impl_Host_Device Class.
   * This class inherits from the Controller_Impl and it aims at implementing the "loadDeviceState" and "loadDeviceParam"
   * methods which copy the State and Param objects from the host backend to the device backend in the Host-Device mode
   * of the framework. This class is the one implemented in the client code.
   */
  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    class Controller_Impl_Host_Device
    : public Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model> {

      public:

        /** Controller_Impl_Host_Only Constructor. This constructor is used in the client code.*/
        Controller_Impl_Host_Device(MetaParam mp_) :
          Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>(mp_){}

        /** Copies the State object from the host backend to the device backend .*/
        void loadDeviceState();

        /** Copies the Param object from the host backend to the device backend .*/
        void loadDeviceParam();

        State getState();

         using Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::stateLock;
         using Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::ctlMetaParam;

    };


  //
  // Controller definition:
  //

  // Constructor/Destructor:
  template<typename MetaParam, typename Param, typename State>
    Controller<MetaParam, Param, State>::Controller()
    {
      // Other public data:
      SDL_AtomicSet(&modelTBegin, 0);
      SDL_AtomicSet(&modelTEnd, 0);

      // Data control initialisations:
      SDL_AtomicSet(&modelPause, 1);              // Start in pause. modelRunning = true; --> in modelExecutor.
      SDL_AtomicSet(&modelOver, 0);
      modelExecAccessLoop   = false;
      SDL_AtomicSet(&modelRunning, 1);
      SDL_AtomicSet(&paramUpdated, 0);
      SDL_AtomicSet(&stateUpdated, 0);

    }

  template<typename MetaParam, typename Param, typename State>
    Controller<MetaParam, Param, State>::~Controller() throw (){ };

  //
  // Public API:
  //

  // Step API:
  template<typename MetaParam, typename Param, typename State>
    void Controller<MetaParam, Param, State>::setTimeInterval(int timeBegin, int timeEnd)
    {
      SDL_AtomicSet(&modelTBegin, timeBegin);
      SDL_AtomicSet(&modelTEnd, timeEnd);
    }

  // Event API:

  // TODO WARNING : start seulement quand en pause sinon modèle accède pendant le swap !!!!
  // Retour de start == 0 si vérif OK, code d'erreur sinon.
  template<typename MetaParam, typename Param, typename State>
    int Controller<MetaParam, Param, State>::start()
    {
      // Précond : modelPause == true
      assert( SDL_AtomicGet(&modelPause) == 1 || SDL_AtomicGet(&modelOver) == 1);

      //un code de verification pour tester si d'autres conditions (e.g. timestep courrant) sont remplies:
      //if(condition...){return verifCode;}
      int verifCode = 0;

      //swap pointers if param/state objects have been updated
      if(SDL_AtomicGet(&paramUpdated) == 1){
        Debug("\nController.start(): Param have been updated. Acquire locks and swap pointer.");

        Autolock v_param(paramLock);

        // paramLock.lock();
        swapParamPointer();
        loadDeviceParam();
        // paramLock.unlock();

        SDL_AtomicSet(&paramUpdated, 0);
        Debug("\nController.start(): Unlock param...  done");
      }

      if(SDL_AtomicGet(&stateUpdated) == 1){
        Debug("\nController.start(): State have been updated. Acquire locks and swap pointer.");

        Autolock v_state(stateLock);

        // stateLock.lock();
        swapStatePointer();
        loadDeviceState();
        // stateLock.unlock();

        SDL_AtomicSet(&stateUpdated, 0);
        Debug("\nController.start(): Unlock state...  done");
      }

      // WakeUP at modelTBegin:
      SDL_AtomicSet( &modelPause, 0);
      SDL_AtomicSet( &modelOver, 0);  //version c++03...
      modelT = (int)SDL_AtomicGet(&modelTBegin);
      modelExecAccessLoop = true;
      Debug("Controller.start(): unlock condition variable of model executor (notify_one)");
      // modelExecCV.notify_one();
      Autolock v(modelExecLock);
      modelExecCV.signal();

      // Return:
      return verifCode;
    }

  template<typename MetaParam, typename Param, typename State>
    void Controller<MetaParam, Param, State>::stopAndJoin()
    {
      Debug("\nController.stopAndJoin(): set modelPause=true and modelRunning=false for model executor (notify_one)");
      SDL_AtomicSet( &modelPause, 1);
      SDL_AtomicSet(&modelRunning, 0);
      modelExecAccessLoop = true;
      // modelExecCV.notify_one();
      Autolock v(modelExecLock);
      modelExecCV.signal();

      // modelExecutor->join();     //version c++03, plus besoin ????
      Debug("\nController.stopAndJoin(): join model executor done.");
    }

  template<typename MetaParam, typename Param, typename State>
    void Controller<MetaParam, Param, State>::playPause()
    {
      if(SDL_AtomicGet( &modelPause ) == 1)
      {
        SDL_AtomicSet( &modelPause, 0);
        modelExecAccessLoop = true;
        Autolock v(modelExecLock);
        modelExecCV.signal();
        Debug("\nController.start(): set modelPause to false for model executor (notify_one)")
      }
      else
      {
        SDL_AtomicSet( &modelPause, 1);
        Debug("\nController.start(): set modelPause to true for model executor (notify_one)")
      }
      // modelExecCV.notify_one();

    }

  // Interrogation API:

  template<typename MetaParam, typename Param, typename State>
    bool Controller<MetaParam, Param, State>::isPause() { return (SDL_AtomicGet( &modelPause) == 1); }

  template<typename MetaParam, typename Param, typename State>
    bool Controller<MetaParam, Param, State>::isOver() { return (SDL_AtomicGet(&modelOver) == 1); }

  template<typename MetaParam, typename Param, typename State>
    int Controller<MetaParam, Param, State>::getCurrentTimeStep() { return modelT; }


  //
  // Controller_Impl definitions:
  //

  // Constructor/Destructor:
  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::Controller_Impl(MetaParam mp)
    :
      ctlMetaParam(mp),
      // ctlMetaParamDevice(mp),            //create metaparam object on device
      ctlState1(&ctlMetaParam),
      ctlState2(&ctlMetaParam),
      ctlParamDevice(),                  //create param object on device
      ctlStateDevice(&ctlMetaParam)     //create state object on device
  {

    // Data Pointer initialisations:
    ctlParamPublic  = &ctlParam1;
    ctlParamIntern  = &ctlParam2;
    ctlStatePublic  = &ctlState1;
    ctlStateIntern  = &ctlState2;

    // Create a new model executor that will launch a new thread:
    modelExecutor = new ModelExecutor_Impl(this);

  }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::~Controller_Impl() throw (){
      delete modelExecutor;
    };

  // Configuration API:

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::setParam(Param* p){
      Debug("\nController_Impl.setParam(): acquire lock, copy to controller buffer, release lock...");
      // paramLock.lock();
      Autolock v(paramLock);
      // copy de l'objet pointé par p dans la zone pointé par ctlParamPublic
      //copy constructor -> automatique si std::vector...
      *ctlParamPublic = *p;
      // paramLock.unlock();

      // swapParamPointer();
      // loadDeviceParam();

      SDL_AtomicSet(&paramUpdated, 1);
      Debug("\nController_Impl.setParam(): done.");
    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::setState(State* s){
      Debug("\nController_Impl.setState(): acquire lock, copy to controller buffer, release lock...");
      // stateLock.lock();
      Autolock v(stateLock);
      //copy constructor
      (*ctlStatePublic)=(*s);

      // swapStatePointer();
      // loadDeviceState();

      // stateLock.unlock();
      SDL_AtomicSet(&stateUpdated, 1);
      Debug("\nController_Impl.setState(): done.");
    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::setConsumers(std::vector<Consumer*> cons)
    {

      consumers.clear();

      // Load consumer pointers
      for(int i=0; i<(int)cons.size(); ++i){
        consumers.push_back(cons[i]);
      }

      // Clear & clean producers:
      for(int i=0; i<(int)producers.size(); ++i){
        delete producers[i];
        delete brokers[i];
      }

      producers.clear();
      brokers.clear();

      // Create producer configuration map
      std::map< ProducerFactory*, std::set<int> > producerConfigMap;

      // Create broker configuration map
      std::map< ProducerFactory*, std::map < Consumer*, std::set<int> > > brokerConfigMap;

      // Create consumer configuration map
      std::map< ProducerFactory*, Broker*> consumerConfigMap;

      // Adding producer factories to the map.
      // std::map guarantee the unicity of each specific factory
      for(int i = 0; i<(int)consumers.size(); ++i){

        // Get parameters:
        std::vector<ProducerParameter> producerParameters = consumers[i]->getProducerParameters();

        for(int j = 0; j<(int)producerParameters.size(); ++j){

          producerConfigMap[ producerParameters[j].prFact ].insert(producerParameters[j].period);

          brokerConfigMap[ producerParameters[j].prFact ][consumers[i]].insert(producerParameters[j].period);
        }

      }

      // Create producers according to the map
      std::map< ProducerFactory*, std::set<int> >::iterator it = producerConfigMap.begin();
      std::map< ProducerFactory*, std::map < Consumer*, std::set<int> > >::iterator itb = brokerConfigMap.begin();

      for( ; it != producerConfigMap.end(); it++, itb++){

        //producers are created via the build method of the producerFactories
        producers.push_back( static_cast<Producer<MetaParam, State, StateDevice>*>(it->first->build()) );

        //mandatary configuration method : producers must receive their config.
        producers.back()->setConfig(ProducerConfig(it->second), &ctlMetaParam);

        //brokers are created
        brokers.push_back(
            new Broker(
              BrokerConfig(itb->second),
              producers.back()->getRequestedBufferSize(),
              it->first
              )
            );

        //associate broker to producer
        producers.back()->setBroker(brokers.back());

        //fill consumerconfigmap
        consumerConfigMap[ (ProducerFactory*)(it->first) ] = brokers.back();

      }

      //set consumers configs
      for(int i = 0; i<(int)consumers.size(); ++i){
        consumers[i]->setConfig(consumerConfigMap);
      }

    }

  // Internal API:
  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::swapStatePointer()
    {
      State * tmpS = ctlStatePublic;
      ctlStatePublic = ctlStateIntern;
      ctlStateIntern = tmpS;
    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::swapParamPointer()
    {
      Param * tmpP = ctlParamPublic;
      ctlParamPublic = ctlParamIntern;
      ctlParamIntern = tmpP;
    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl_Host_Device<MetaParam, Param, ParamDevice, State, StateDevice, Model>::loadDeviceState()
    {
      (this->ctlStateDevice).template copy<State>(*(this->ctlStateIntern));
    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl_Host_Device<MetaParam, Param, ParamDevice, State, StateDevice, Model>::loadDeviceParam()
    {
      (this->ctlParamDevice).copyFromHost(*(this->ctlParamIntern));
    }



  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    State Controller_Impl_Host_Device<MetaParam, Param, ParamDevice, State, StateDevice, Model>::getState(){
      
      Autolock v(stateLock);
      State s(&ctlMetaParam);
      s.template copy<StateDevice>(this->ctlStateDevice);
      return s;
    }

  template<typename MetaParam, typename Param, typename State, typename Model>
    void Controller_Impl_Host_Only<MetaParam, Param, State, Model>::loadDeviceState(){}

  template<typename MetaParam, typename Param, typename State, typename Model>
    void Controller_Impl_Host_Only<MetaParam, Param, State, Model>::loadDeviceParam(){}


  template<typename MetaParam, typename Param, typename State, typename Model>
    State Controller_Impl_Host_Only<MetaParam, Param, State, Model>::getState(){
      
      Autolock v(stateLock);
      return ctlStateIntern;
    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::executeProducers(){

      for(int i=0; i<(int)producers.size(); ++i){
        producers[i]->execute(ctlStateIntern, &ctlStateDevice, modelT, (int)SDL_AtomicGet(&modelTBegin));
      }

    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::sendErrorToConsumers(int error_code, int step){

      for(int i=0; i<(int)consumers.size(); ++i){
        consumers[i]->notifyError(error_code, step);
      }

    }


  //
  // ModelExecutor_Impl:
  //

  // Constructor/Destructor:
  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::ModelExecutor_Impl::ModelExecutor_Impl(Controller_Impl* p)
    :controller_i(p) {

      executorThread = SDL_CreateThread(threadLoop, "threadLoop", this);

      // C++11 version
      // executorThread = new std::thread(&ModelExecutor_Impl::threadLoop, this);
    }

  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::ModelExecutor_Impl::~ModelExecutor_Impl() throw ()
    {
      // SDL_KillThread(executorThread); // TODO : how to delete the SDL executorThread

      // C++11 version
      // delete executorThread;
    }

  // API:
  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    void Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::ModelExecutor_Impl::join() {

      int status;
      SDL_WaitThread(executorThread, &status);

      // C++11 version
      // executorThread->join();
    }

  // Operational:
  template<typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice, typename Model>
    int Controller_Impl<MetaParam, Param, ParamDevice, State, StateDevice, Model>::ModelExecutor_Impl::threadLoop(void * ptr)
    {

      Controller_Impl* controller_i = ((ModelExecutor_Impl *)ptr)->controller_i;

      Debug("\nModelExecutor_Impl.startThread: assert modelRunning\n")
        assert( SDL_AtomicGet(&(controller_i->modelRunning)) == 1);

      while(SDL_AtomicGet(&(controller_i->modelRunning)) == 1)
      {
        Debug("\nModelExecutor_Impl.startThread: wait for condition variable to be notified by controller_i and test predicate")

        Autolock v(controller_i->modelExecLock);
        if(!(controller_i->modelExecAccessLoop)){ (controller_i->modelExecCV).wait(controller_i->modelExecLock); }
        //C++11
        // (controller_i->modelExecCV).wait(controller_i->modelExecLock, [this]{return controller_i->modelExecAccessLoop;});

        Debug("\nModelExecutor_Impl.startThread: condition variable unlocked")
        controller_i->modelExecAccessLoop = false;

        //to record simulation time
        struct timeval begin;
        gettimeofday(&begin, NULL);

        while( SDL_AtomicGet(&controller_i->modelOver) == 0 && SDL_AtomicGet( &(controller_i->modelPause)) == 0 )
          // C++11
          // while(  !controller_i->modelPause )
        {
          Debug("\nModelExecutor_Impl.startThread: another step in the model loop")

          // ALGO STEP 1
          int error_code = ModelBase<Model, MetaParam, Param, ParamDevice, State, StateDevice>::callAlgoStep1(
              &(controller_i->ctlMetaParam),
              controller_i->ctlParamIntern,
              &(controller_i->ctlParamDevice),
              controller_i->ctlStateIntern,
              &(controller_i->ctlStateDevice),
              controller_i->modelT
              );

          if(error_code != 0){
            std::cout << "Simulation stopped because an error occured : " << error_code << std::endl;
            controller_i->sendErrorToConsumers(error_code, controller_i->modelT);
          }

          // Call Producers:
          controller_i->executeProducers();

          // ALGO STEP 2
          int error_code2 = ModelBase<Model, MetaParam, Param, ParamDevice, State, StateDevice>::callAlgoStep2(
              &(controller_i->ctlMetaParam),
              controller_i->ctlParamIntern,
              &(controller_i->ctlParamDevice),
              controller_i->ctlStateIntern,
              &(controller_i->ctlStateDevice),
              controller_i->modelT
              );

          if(error_code2 != 0){
            std::cout << "Simulation stopped because an error occured : " << error_code2 << std::endl;
            controller_i->sendErrorToConsumers(error_code2, controller_i->modelT);
          }

          if( (error_code != 0)   ||
              (error_code2 != 0)  ||
              controller_i->modelT >= (int)SDL_AtomicGet(&(controller_i->modelTEnd)) )
          {
            //display simulation time
            struct timeval end;
            gettimeofday(&end, NULL);
            std::cout << "Simu time: " << 1000000 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) << std::endl;
            //
            SDL_AtomicSet(&controller_i->modelOver, 1);
          }
          else{ SDL_AtomicSet(&controller_i->modelOver, 0); }

          //slow down simu
          // usleep(1000000); //in microseconds
          // C++11
          // std::chrono::milliseconds dura( 200 );
          // std::this_thread::sleep_for( dura );

          controller_i->modelT++;
        }

      }
      
      // std::cout << "Fin ModelExecutor_Impl\n" << std::endl;

      return 0;
    }
} // End namespace

#endif
