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

#ifndef _CUSTOMDISPLAY_
#define _CUSTOMDISPLAY_

#include "custom_objects.hpp"
#include "glwidget.hpp"
#include "producer.hpp"

namespace mg{

  void printcustom(EmbryoState *mg_buf, GLWidget * glw);
  void printcustom2D(EmbryoState *mg_buf, float *modelview, float *projection, float w, float h, GLWidget * glw);
}
 
#define PRINTCUSTOM  	printcustom(mg_buf, this);
#define PRINTCUSTOM2D  	printcustom2D(mg_buf, modelview, projection, width(), height(), this);

#endif
