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

#ifndef _CCREATOR_H
#define _CCREATOR_H

// Model includes:
#include "metaparam.hpp"
#include "model.hpp"
#include "param_host.hpp"
#include "param_device.hpp"
#include "state_host.hpp"
#include "state_device.hpp"

// Thrust objects:
#include "thrust_objects.hpp"

// ISF includes:
#include "controller.hpp"

namespace mg{

  isf::Controller<
  					MetaParam<HOST>, Param_Host, State_Host
  				>* createNewController(MetaParam<HOST> meta);

}

#endif
