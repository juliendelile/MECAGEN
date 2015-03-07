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

#ifndef CELLS_QAPPLICATION_H
#define CELLS_QAPPLICATION_H


// OpenGL:
#include "GL/glew.h"
#include <GL/glut.h>


// Qt includes
#include <QApplication>
#include <QDesktopWidget>

// Project includes
// ISF include:
#include "controller.hpp"
// Thrust:
#include "thrust_objects.hpp"
// Model/simulation
#include "metaparam.hpp"
// #include "param.hpp"
#include "param_host.hpp"
#include "state_host.hpp"
#include "model.hpp"
// GUI:
#include "window.hpp"
#include "cells_visualizer.hpp"

namespace mg {

  // A consumer that log results:
  class FileLogger;

  class CellsQApplication : public QApplication
  {

    public:
      CellsQApplication(int & argc, char** argv);
      virtual ~CellsQApplication() throw ();

      bool is_init_ok(){ return init_ok;}

      bool init_ok;

    private:
      // Model/Simulation:
      MetaParam<HOST>                                               *metaParam;
      Param_Host                                                    *param;
      State_Host                                                    *state;
      isf::Controller< MetaParam<HOST>, Param_Host, State_Host >    *controller;

      // File logger:
      FileLogger *    filelogger;
      char            adress_filelogger[300];

      // Visualization:
      Window      *window;
      VizConsumer *vizconsumer;
  };
}

#endif
