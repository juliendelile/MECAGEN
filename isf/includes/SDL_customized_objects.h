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

#ifndef _SDLCUSTOMIZEDOBJECTS_H_
#define _SDLCUSTOMIZEDOBJECTS_H_

//Include SDL :
#include "SDL.h"

// see http://h-deb.clg.qc.ca/Sujets/Client-Serveur/Mutex-Autolock.html for explanation (french)

namespace isf {


  //Rend un objet Uncopyable:
  class Uncopyable
  {
    private:
      //Bloquer la copie
      Uncopyable(const Uncopyable &);
      Uncopyable & operator=(const Uncopyable &);

    protected:
      // Permettre l'instanciation des enfants!
      Uncopyable(){}
      virtual ~Uncopyable() throw(){}
  };

  /***
   * Outils de synchronisations
   */

  class Mutex : public Uncopyable
  {
    friend class ConditionVariable;

    private :
    SDL_mutex *mutex;

    public :
    Mutex() : mutex(SDL_CreateMutex()) { }
    ~Mutex() throw () { SDL_DestroyMutex(mutex); }

    int P() throw () { return SDL_mutexP(mutex); }
    int V() throw () { return SDL_mutexV(mutex); }
  };

  class Autolock
  {
    private :
      Mutex &mutex;

    public :
      Autolock(Mutex &mutex_) throw() : mutex(mutex_) { mutex.P(); }
      ~Autolock() throw () { mutex.V(); }
  };

  class Semaphore
  {
    private :
      SDL_sem *semaphore;

    public :
      Semaphore(Uint32 init) : semaphore(SDL_CreateSemaphore(init)) { }
      ~Semaphore() throw() { SDL_DestroySemaphore(semaphore); }

      int wait() throw () { return SDL_SemWait(semaphore); }
      int tryWait() throw() { return SDL_SemTryWait(semaphore); }
      int waitTimeOut(Uint32 timeout) throw() { return SDL_SemWaitTimeout(semaphore, timeout); }
      int post() throw () { return SDL_SemPost(semaphore); }
      int value() throw () { return SDL_SemValue(semaphore); }
  };

  class ConditionVariable
  {
    private :
      SDL_cond* condition;
      SDL_mutex *mutex;

    public :
      ConditionVariable() : condition(SDL_CreateCond()), mutex(SDL_CreateMutex()) { SDL_mutexP(mutex); }
      ~ConditionVariable() throw() { SDL_DestroyCond(condition); }

      int signal() { return SDL_CondSignal(condition); }
      int broadcast() { return SDL_CondBroadcast(condition); }
      int wait() { return SDL_CondWait(condition, mutex); }
      int wait(Mutex & m) { return SDL_CondWait(condition, m.mutex); }
  };


}

#endif
