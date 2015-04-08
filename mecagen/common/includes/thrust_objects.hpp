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

#ifndef _THRUST_OBJECTS_H_
#define _THRUST_OBJECTS_H_

#include "serialization.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h> 

#include "nvVector.h"

namespace mg {

  typedef unsigned int              uint;

  typedef nv::vec4<double>          d4;
  typedef nv::vec3<double>          d3;
  typedef nv::vec2<double>          d2;
  typedef nv::vec3<unsigned int>    u3;
  typedef nv::vec3<int>             i3;
  typedef nv::vec3<float>           f3;


  /** Definition of two backend indicator used to template the Param, MetaParam and State objects. 
   * The backend characterization is given by the name.
   */
  enum { HOST, DEVICE };

  /** Backend template specialization mechanism.
   * According to the backend indicator, corresponding types of thrust vectors are defined with the same 
   * denomination
   */
  template<int integer>
    struct Backend;

  /** Device backend types specialization. */
  template<>
    struct Backend<DEVICE>
    {

      typedef typename thrust::device_vector<double>      vecDouble;
      typedef typename thrust::device_vector<float>       vecFloat;
      typedef typename thrust::device_vector<int>         vecInt;
      typedef typename thrust::device_vector<uint>        vecUint;
      typedef typename thrust::device_vector<d4>          vecD4;
      typedef typename thrust::device_vector<d3>          vecD3;
      typedef typename thrust::device_vector<d2>          vecD2;
    };

  /** Host backend types specialization. */
  template<>
    struct Backend<HOST>
    {
      typedef typename thrust::host_vector<double>        vecDouble;
      typedef typename thrust::host_vector<float>         vecFloat;
      typedef typename thrust::host_vector<int>           vecInt;
      typedef typename thrust::host_vector<uint>          vecUint;
      typedef typename thrust::host_vector<d4>            vecD4;
      typedef typename thrust::host_vector<d3>            vecD3;
      typedef typename thrust::host_vector<d2>            vecD2;
    };

  // Define a generic atomic add function 
  // It returns the old value stored at the adress "adress"
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  inline __device__ uint mgAtomicAddOne(uint * adress){
    return atomicAdd(adress, 1);
  };
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
  inline uint mgAtomicAddOne(uint * adress){
    ++(*adress);
    return ((*adress)-1);
  };
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
  inline uint mgAtomicAddOne(uint * adress){
  
  // #pragma omp atomic
  //   ++(*adress);
  //   //si un autre thread OMP modify la valeur juste après, la ligne suivante est fausse non ? Matthieu ?
  //   return ((*adress)-1);
  // };

    #ifdef __GNUC__
      // #error "__GNUC__" 
      return __sync_fetch_and_add(adress, 1);
    #elif defined(_OPENMP) and _OPENMP>=201107
      // #error "OPENMP 2.1" 
      uint t;
      #pragma omp atomic capture
      { t = *adress; *adress += 1; }
      return t;
    #else 
      #error "Requires gcc or OpenMP>=3.1" 
    #endif

  };
#endif

  //This functor is used to generate the random values at State object initialization
  __host__ static __inline__ double rand_01()
  {
    return ((double)rand()/RAND_MAX);
  }


}


#define BOOST_SAVE_THRUST_HOST_VECTOR(T)                                \
namespace boost { namespace serialization {                             \
template<class Archive>                                                 \
void save(                                                              \
        Archive & ar,                                                   \
        const thrust::host_vector< T > & thrust_vec,                    \
        const unsigned int version                                      \
){                                                                      \
  std::vector< T > stl_vec(thrust_vec.size());                          \
  thrust::copy(thrust_vec.begin(), thrust_vec.end(), stl_vec.begin());  \
  ar & BOOST_SERIALIZATION_NVP(stl_vec);                                \
}                                                                       \
}}

#define BOOST_LOAD_THRUST_HOST_VECTOR(T)                                \
namespace boost { namespace serialization {                             \
template<class Archive>                                                 \
void load(                                                              \
        Archive & ar,                                                   \
        thrust::host_vector< T > & thrust_vec,                          \
        unsigned int version                                            \
){                                                                      \
  std::vector< T > stl_vec;                                             \
  ar & BOOST_SERIALIZATION_NVP(stl_vec);                                \
  thrust::copy(stl_vec.begin(), stl_vec.end(), thrust_vec.begin());     \
}                                                                       \
}}

BOOST_SAVE_THRUST_HOST_VECTOR(double)
BOOST_LOAD_THRUST_HOST_VECTOR(double)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<double>)

BOOST_SAVE_THRUST_HOST_VECTOR(float)
BOOST_LOAD_THRUST_HOST_VECTOR(float)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<float>)

BOOST_SAVE_THRUST_HOST_VECTOR(int)
BOOST_LOAD_THRUST_HOST_VECTOR(int)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<int>)

BOOST_SAVE_THRUST_HOST_VECTOR(uint)
BOOST_LOAD_THRUST_HOST_VECTOR(uint)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<uint>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::d2)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::d2)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::d2>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::d3)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::d3)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::d3>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::d4)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::d4)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::d4>)

#endif
