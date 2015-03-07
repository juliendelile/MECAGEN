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

#ifndef _TOOLS_HPP_2013_08
#define _TOOLS_HPP_2013_08

// Standard output:
#include <iostream>

//#define NDEBUG  //remove comment on NDEBUG to remove assertion in the code

// #define DEBUG_BUILD      //uncomment to print debug messages

#ifdef DEBUG_BUILD
  #define Debug(x)  std::cout << x << std::endl;
#else
  #define Debug(x)
#endif

#endif
