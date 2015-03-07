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

#ifndef DISPLAYPARAMETERS_H
#define DISPLAYPARAMETERS_H

#include "thrust_objects.hpp"
#include "define.hpp"

namespace mg{

   struct DisplayParams{
    
      float                 typeColorR[4];
      float                 typeColorG[4];
      float                 typeColorB[4];      
      int                   colorByType;
      float                 proteinColorR[NUMPROTEINmax];
      float                 proteinColorG[NUMPROTEINmax];
      float                 proteinColorB[NUMPROTEINmax];
      float                 proteinThreshold[NUMPROTEINmax];
      int                   currentProtein;
      float                 ligandColorR[NUMLIGmax];
      float                 ligandColorG[NUMLIGmax];
      float                 ligandColorB[NUMLIGmax];
      float                 ligandThreshold[NUMLIGmax];
      int                   currentLigand;
      float                 noneColorR;
      float                 noneColorG;
      float                 noneColorB;
      int                   drawAxes;
      int                   candAxesId;
      int                   axeAB;
      int                   axe1;

  };

}

#endif