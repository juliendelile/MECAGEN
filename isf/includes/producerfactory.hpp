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

#ifndef _FACTORY_H_2013_09
#define _FACTORY_H_2013_09

// Standard C definition (size_t):
#include <cstddef>
#include <iostream>

namespace isf {

  // Top level factory:
  /** ProducerFactory class.
   * The ProducerFactory is the top-level interface and parent class of the ProducerFactory_Impl class.
   */
  class ProducerFactory{
    public:
      // API:
      /** Pure virtual function implemented in the ProducerFactory_Impl class.*/
      virtual void* build() = 0;

    protected:
      /** ProducerFactory constructor.*/
      ProducerFactory(){};

      /** ProducerFactory destructor.*/
      virtual ~ProducerFactory() throw (){};
  };

  /** ProducerFactor_Impl class.
   * The ProducerFactor_Impl is the actual implementation of the ProducerFactory concept. It inherits
   * the ProducerFactory class and is used to call the constructor of the client-specified producer method
   * class. As this class is also used to identify the different producers and brokers in the ISF framework,
   * this class is rendered uncopyable and assure singles instancing following the
   * <a href="http://en.wikipedia.org/wiki/Singleton_pattern">singleton pattern</a>.
   */
  template<typename PChild>
    class ProducerFactory_Impl : public ProducerFactory {
      public:
        // API:
        /** Calls the PChild constructor.*/
        virtual void * build();

        // Singleton interfece:
        /** Implements the singleton pattern.*/
        static ProducerFactory_Impl* getInstance()
        {
          static ProducerFactory_Impl instance;
          return &instance;
        }

      private:

        /** ProducerFactory_Impl constructor. */
        ProducerFactory_Impl(){  };

        /** ProducerFactory_Impl destructor. */
        virtual ~ProducerFactory_Impl() throw (){};

        /** Is required to make the object uncopiable. Do not implement!*/
        ProducerFactory_Impl(ProducerFactory_Impl const&);

        /** Is required to make the object uncopiable. Do not implement!*/
        void operator = (ProducerFactory_Impl const&);

    };

  //
  // Definition:
  //
  template<typename PChild>
    void* ProducerFactory_Impl<PChild>::build(){
      return new PChild();
    }

} // End namespace
#endif
