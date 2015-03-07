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

#ifndef _EVENTMANAGER_H_2012_08
#define _EVENTMANAGER_H_2012_08


#include "SDL_customized_objects.h"

// Standard thread:
// #include <mutex>
// #include <condition_variable>


// Standard collection:
#include <deque>
#include <cstdio>

namespace isf {

  /** Event class.
  * An Event is a template object which contains a target object to notify and a pointer to one of its 
  * internal method to call when the event is processed.
  */
  template<typename TargetT>
    class Event
    {
      public:
        typedef void (TargetT::*ActionT)();

        /** Default Event constructor.*/ 
        Event( ) { }

        /** Overloaded Event constructor. 
        * @param[in] target_    The pointer of the TargetT class the event's action will act on.
        * @param[in] action_    The pointer of the method of the TargetT class that is called by the process()
        * method.
        */
        Event(TargetT* target_, ActionT action_) :
          target(target_), action(action_) { }

        /** Event destructor.*/
        virtual ~Event() throw () { }

        /** Call the method specified in the event.
        */
        virtual void process() {
          (target ->* action)();
        }

      private:
        /** Is used to store the pointer of the TargetT class the event's action will act on.*/
        TargetT*  target;
        /** Is used to store the pointer of the method of the TargetT class that is called by the process()*/
        ActionT   action;
    };

  /** ParametrizedEvent class.
  * A ParametrizedEvent is a customized Event which contains arguments that are passed to the Event action's
  * method when processed.
  */
  template<typename TargetT, typename ArgT>
    class ParametrizedEvent:
      public Event<TargetT>
  {
    public:
      typedef void (TargetT::*ActionT)(ArgT);

      /** Default ParametrizedEvent constructor.*/
      ParametrizedEvent(){}

      /** Overloaded ParametrizedEvent constructor. 
        * @param[in] target_    The pointer of the TargetT class the event's action will act on.
        * @param[in] action_    The pointer of the method of the TargetT class that is called by the process()
        * method.
        * @param[in] arg_       The argument object passed to the action method.
        */
      ParametrizedEvent(TargetT* target_, ActionT action_, ArgT arg_) :
        target(target_), action(action_), arg(arg_) { }

      /** ParametrizedEvent destructor.*/
      virtual ~ParametrizedEvent() throw () { }

      /** Call the method specified in the event.
        */
      void process() {
        // std::cout << "parametrized event is processed " << std::endl;
        (target ->* action)(arg);
      }

    private:
      /** Is used to store the pointer of the TargetT class the event's action will act on.*/
      TargetT*  target;
      /** Is used to store the pointer of the method of the TargetT class that is called by the process()*/
      ActionT   action;
      /** Is used to store the argument object passed to the action method. Typically, a NotifyParameten object.*/
      ArgT      arg;
  };


/** EventManager class.
* The EventManager class is a templated class which contains the synchronization mechanism designed in ISF.
* It contains a internal stack of event: a std::deque< Event<T>* > that is filled by the notifying thread via
* the send() method and read by the event manager's owner thread via the read() method. The synchronization is 
* assured by a std::mutex variable that the sending and reading threads acquire and release when they use the 
* deque.
*/
template<typename T>
    class EventManager
    {
      public :

        /** EventManager constructor.*/
        EventManager() {}
        // EventManager():dowait(true) {}

        /** EventManager destructor.*/
        virtual ~EventManager() throw() {}

        /** Push an event at the back of the deque. This method is called by the notifying thread.*/
        void send(Event<T>* event)
        {
          // The lock guard acquires the mutex when created (i.e. mutex.lock)
          // and realases it (i.e. mutex.unlock) when being destroyed, i.e. when leaving scope.
          
          

          // C++03 / SDL version
          Autolock v(mutex);                //Accès en mutex ;
          fifo.push_back(event);   //Ajout de l'événement dans la fifo ;
          // dowait = false;                     //Au moins un événement : pas d'attente après le swap !
          cv.signal();                      //Réveiller thread si était en attente.

          // printf("fifo %ld size after send : %d\n",(long)this, (int)fifo.size());

          // C++11 version
          // std::lock_guard<std::mutex> lk(mutex_);
          // fifo.push_back(event);
          // cv.notify_one();
        }

        /** Pop an event at the front of the deque. This method is called by the receiver thread.*/
        Event<T>* read()
        {
          // The unique lock works like the guard but also allows cv.wait to release the mutex when waiting and
          // to reacquire it when being notified.
        
          // C++03 / SDL version
          Autolock v(mutex);                  //Accès en mutex ;
          // if(dowait) { cv.wait(mutex); }    //Se bloquer + libérer le mutex si attente demandée.
          if(fifo.size() == 0) { cv.wait(mutex); }   
          // dowait=true;
          Event<T>* ret = fifo.front();
          fifo.pop_front();
          
          // printf("fifo %ld size after read : %d\n",(long)this, (int)fifo.size());
          
          // C++11 version
          // std::unique_lock<std::mutex> lk(mutex_);
          // cv.wait(lk, [this]{ return this->fifo.size() > 0;});
          // Event<T>* ret = fifo.front();
          // fifo.pop_front();

          return ret;
        }

      private:
        /** Is used to synchronize the deque processing actions.*/
        ConditionVariable cv;
        // bool dowait;        

        // C++11 version
        // std::condition_variable cv;

        /** Is used to synchronize the deque processing actions.*/
        Mutex mutex;

        // C++11 version
        // std::mutex mutex_;

        /** Is used to stack the incoming events. Events are removed in the read() method.*/
        std::deque< Event<T>* > fifo;
    };

} // End namespace

#endif
