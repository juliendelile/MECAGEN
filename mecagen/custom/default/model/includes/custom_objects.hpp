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

#ifndef _CUSTOMOBJECTS_
#define _CUSTOMOBJECTS_

#include "thrust_objects.hpp"
#include "define.hpp"

namespace mg {

  struct CustomParams{

    uint    bigarray[2000];
    uint    bigarray2[1000];

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
        ar
          & BOOST_SERIALIZATION_NVP(bigarray)
          & BOOST_SERIALIZATION_NVP(bigarray2)
          ;
      }
  };

  struct CustomState{

    d3    somevalues[NUMCELLmax];

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
        ar
          & BOOST_SERIALIZATION_NVP(somevalues)
          ;
      }
  };

  template< int T >
  struct CustomStateTemplate{

    typename Backend<T>::vecD3        somevalues;

    // Constructor
    CustomStateTemplate():somevalues(NUMCELLmax){}

    CustomStateTemplate& copy(const CustomStateTemplate<HOST> & other){
      somevalues = other.somevalues;
      return *this;
    }

    CustomStateTemplate& copy(const CustomStateTemplate<DEVICE> & other){
      somevalues = other.somevalues;
      return *this;
    }

    // Serialization 
    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
        ar
          & BOOST_SERIALIZATION_NVP(somevalues);
      }
  }; 

  /** Backend template specialization mechanism.
   * According to the backend indicator, corresponding types of thrust vectors are defined with the same
   * denomination
   */
  template<int integer>
    struct BackendCustom;

  /** Device backend types specialization. */
  template<>
    struct BackendCustom<DEVICE>
    {
      typedef typename thrust::device_vector<CustomParams>                vecCustomParam;
      typedef CustomStateTemplate<DEVICE>                                 CustomState;
    };

  /** Host backend types specialization. */
  template<>
    struct BackendCustom<HOST>
    {
      typedef typename thrust::host_vector<CustomParams>                  vecCustomParam;
      typedef CustomStateTemplate<HOST>                                   CustomState;
    };

  struct CustomStateBuffer{

    d3      somevalues[NUMCELLmax];

    void copy(BackendCustom<DEVICE>::CustomState *sca, uint numCells){
      thrust::copy(sca->somevalues.begin(), sca->somevalues.end(), somevalues);
    }
  };
}

BOOST_SAVE_THRUST_HOST_VECTOR(mg::CustomParams)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::CustomParams)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::CustomParams>)

#endif
