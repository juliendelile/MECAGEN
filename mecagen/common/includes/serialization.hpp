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

#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <fstream>   //ofstream

// Boost -- serialization
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>


// Boost -- serialization
#include <boost/serialization/string.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/split_free.hpp>


//
// Boost serialization
//

template<typename T>
void save(const T& myObject, const char* fname){
  // Create the xml archive file:
  std::ofstream file(fname);
  boost::archive::xml_oarchive oa(file);
  // Dump the parameters:
  oa & BOOST_SERIALIZATION_NVP(myObject);
}

template<typename T>
void load(T& myObject, const char* fname){
  // Open the xml archive file:
  std::ifstream file(fname);
  boost::archive::xml_iarchive ia(file);
  // Read from the archive:
  ia & BOOST_SERIALIZATION_NVP(myObject);
}

#endif
