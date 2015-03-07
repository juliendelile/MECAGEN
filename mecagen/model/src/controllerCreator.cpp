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

#include "controllerCreator.hpp"


namespace mg{

  isf::Controller< MetaParam<HOST>, Param_Host, State_Host >* 
        createNewController(MetaParam<HOST> meta)
  {
    isf::Controller_Impl_Host_Device<
                  MetaParam<HOST>,
                  Param_Host,
                  Param_Device,
                  State_Host,
                  State_Device,
                  Model>* 
        ctl = new isf::Controller_Impl_Host_Device<
                          MetaParam<HOST>,
                          Param_Host,
                          Param_Device,
                          State_Host,
                          State_Device,
                          Model> (meta);

    return ctl;
  }

}
