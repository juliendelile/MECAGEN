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

#ifndef _MODELPROXIES_H_2013_08
#define _MODELPROXIES_H_2013_08

// Include standard:
#include <iostream>

namespace isf{

  /** Parent class of the model proxy mechanism.
   * On one hand, the client may specify Host-Only or Host-Device implementations of the model class, and on the
   * other hand, the Model_Executor_Impl's method "threadloop" must call the client-specified algosteps' methods
   * (algoStep1 and algoStep2) which are defined with a different argument signature according to the host-only or
   * host-device instance of the framework. The model proxy mechanism used here requires to pass the limitation of
   * not having basic virtual static mechanism in C++. This is done by use a templated parent class, with derived
   * class being passed as template to the parent class (see http://stackoverflow.com/questions/6291160/static-virtual-functions-in-c
   * for a reference). This parent class is used by two derived class: ModelHostOnly and ModelHostDevice.
   */
  template< typename ModelDerived, typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice >
                                                                       class ModelBase
  {
    public:

      /** Proxy method called by the Model_Executor_Impl's "threadloop" method. In turn, it calls the ModelBase's derived class'
       * "callAlgoStep1" method.
       */
      static int callAlgoStep1(
          MetaParam * mp,
          Param * p,        ParamDevice * pd,
          State * s,        StateDevice * sd,
          int currentTimeStep)
      {
        return ModelDerived::callAlgoStep1(mp, p, pd, s, sd, currentTimeStep);
      }

      /** Proxy method called by the Model_Executor_Impl's "threadloop" method. Similar to callAlgoStep1.
      */
      static int callAlgoStep2(
          MetaParam * mp,
          Param * p,      ParamDevice * pd,
          State * s,      StateDevice * sd,
          int currentTimeStep)
      {
        return ModelDerived::callAlgoStep2(mp, p, pd, s, sd, currentTimeStep);
      }
  };


  /** Derived class ModelHostOnly of the model proxy mechanism.
   * This class implements the actual call to the client-specified "algoStep1" and "algoStep2" method when the
   * framework is used in Host-Only mode.
   */
  template< typename ModelDerived2, typename MetaParam, typename Param, typename State>
    class ModelHostOnly : public ModelBase<
                          ModelHostOnly<
                          ModelDerived2,
                          MetaParam,
                          Param,
                          State>,
                          MetaParam,
                          Param,
                          Param,
                          State,
                          State >
  {

    public:

      /** Proxy method which passes only the host pointers to the client-specified "algoStep1" method. */
      static int callAlgoStep1(
          MetaParam * mp,
          Param * p,      Param * pd,
          State * s,      State * sd,
          int currentTimeStep)
      {
        return ModelDerived2::algoStep1(mp, p, s, currentTimeStep);
      }

      /** Proxy method which passes only the host pointers to the client-specified "algoStep2" method. */
      static int callAlgoStep2(
          MetaParam * mp,
          Param * p,      Param * pd,
          State * s,      State * sd,
          int currentTimeStep)
      {
        return ModelDerived2::algoStep2(mp, p, s, currentTimeStep);
      }

  };

  /** Derived class ModelHostDevice of the model proxy mechanism.
   * This class implements the actual call to the client-specified "algoStep1" and "algoStep2" method when the
   * framework is used in Host-Device mode.
   */
  template< typename ModelDerived2, typename MetaParam, typename Param, typename ParamDevice, typename State, typename StateDevice>
    class ModelHostDevice : public ModelBase<
                            ModelHostDevice<
                              ModelDerived2,
                              MetaParam,
                              Param,
                              ParamDevice,
                              State,
                              StateDevice>,
                            MetaParam,
                            Param,
                            ParamDevice,
                            State,
                            StateDevice >
  {
    public:

      /** Proxy method which passes all host and device pointers to the client-specified "algoStep1" method. */
      static int callAlgoStep1(
          MetaParam * mp,
          Param * p,      ParamDevice * pd,
          State * s,      StateDevice * sd,
          int currentTimeStep)
      {
        return ModelDerived2::algoStep1(mp, p, pd, s, sd, currentTimeStep);
      }

      /** Proxy method which passes all host and device pointers to the client-specified "algoStep2" method. */
      static int callAlgoStep2(
          MetaParam * mp,
          Param * p,      ParamDevice * pd,
          State * s,      StateDevice * sd,
          int currentTimeStep)
      {
        return ModelDerived2::algoStep2(mp, p, pd, s, sd, currentTimeStep);
      }
  };

}
#endif
