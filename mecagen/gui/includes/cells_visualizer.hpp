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

#ifndef _VIZCONSUMER_HPP_2013_11
#define _VIZCONSUMER_HPP_2013_11

// Project includes
// ISF Class:
#include "consumer.hpp"           // This a consumer !

// Thrust:
#include "thrust_objects.hpp"     // Backend<>
// Model/simulation
#include "param.hpp"
// Consumer visualization:
#include "cells_producer.hpp"
#include "glwidget.hpp"

namespace isf { class ProducerFactory; }

namespace mg {

  // Forwand declarations:
  class GLWidget;

  // Consumer definition:
  class VizConsumer : public isf::Consumer {

    public:
      VizConsumer(GLWidget *_glw);
      virtual ~VizConsumer() throw ();

    protected:
      /** Implements the action of the consumer when it receives a notification.
       * @param[in] buffer   Pointer toward the notifying broker buffer
       * @param[in] prod_id  Id of the notifying broker
       * @param[in] step     Timestep of the notification
       */
      void consumerProcess(void * buffer, isf::ProducerFactory* prod_id, int step);
      void errorManagement(int step, int errorCode){};

    private:
      GLWidget *        glWidget;
  };
};
#endif
