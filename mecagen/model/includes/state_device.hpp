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

#ifndef _STATE_DEVICE_
#define _STATE_DEVICE_

#include "state.hpp"

namespace mg {

  // Forward declarations:
  template<int T> class MetaParam;

  /** State object.
  * This object must be deeply copiable, and follow the rule of three. Here, we use thrust vectors
  * as members so it is indeed the case. However, in the Host-Device mode of the framework, an additional method
  * must be implemented to allow the copy from the backend to the device backend ("copy" method).
  */
  class State_Device: public State<DEVICE>{

    public:

    /** State class constructor.
    * The thrust vectors are initialized in the constructor initializer list via the MetaParam object passed
    * as a parameter.
    */
    State_Device(MetaParam<HOST>* mp):
        State<DEVICE>(mp)
        {}

    /** State class destructor. */
    ~State_Device() throw () {}
  };
}
#endif
