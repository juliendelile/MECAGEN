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

#ifndef _PARAMOBJECTS_H
#define _PARAMOBJECTS_H

// #include "define.hpp"

namespace mg {


  struct CellCycleParams{
    uint mode;
    double param1;
    double param2;
    double param3;
    double param4;
    double volume_ratio;
    uint   mPhaseLength;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(mode)
          & BOOST_SERIALIZATION_NVP(param1)
          & BOOST_SERIALIZATION_NVP(param2)
          & BOOST_SERIALIZATION_NVP(param3)
          & BOOST_SERIALIZATION_NVP(param4)
          & BOOST_SERIALIZATION_NVP(volume_ratio)
          & BOOST_SERIALIZATION_NVP(mPhaseLength)
          ;
      }
  };

  struct LigandParams{
    double diffusion;
    double chi;
    char name[300];

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(diffusion)
          & BOOST_SERIALIZATION_NVP(chi)
          & BOOST_SERIALIZATION_NVP(name)
          ;
      }
  };

  struct PolarizationAxisParams{
    uint idlig;
    uint compMode;
    uint apicoBasalInEpithelium;  // specifies whether the axis is oriented along the apico-basal axis
                                  // or orthogonally to it in epithelial cells
    double param1; // ligand sensibility threshold in cell-cell contact propagation mode

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(idlig)
          & BOOST_SERIALIZATION_NVP(compMode)
          & BOOST_SERIALIZATION_NVP(apicoBasalInEpithelium)
          & BOOST_SERIALIZATION_NVP(param1)
          ;
      }
  };
  
  /** Backend template specialization mechanism for Param structure.
   * According to the backend indicator, corresponding types of thrust vectors are defined with the same 
   * denomination
   */
  template<int integer>
    struct BackendParam;

  /** Device backend types specialization. */
  template<>
    struct BackendParam<DEVICE>
    {
      typedef typename thrust::device_vector<CellCycleParams>            vecCellCycleParams;
      typedef typename thrust::device_vector<LigandParams>               vecLigandParams;
      typedef typename thrust::device_vector<PolarizationAxisParams>     vecPolarizationAxisParams;
    };

  /** Host backend types specialization. */
  template<>
    struct BackendParam<HOST>
    {
      typedef typename thrust::host_vector<CellCycleParams>              vecCellCycleParams;
      typedef typename thrust::host_vector<LigandParams>                 vecLigandParams;
      typedef typename thrust::host_vector<PolarizationAxisParams>       vecPolarizationAxisParams;
    };

} // End namespace

BOOST_SAVE_THRUST_HOST_VECTOR(mg::CellCycleParams)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::CellCycleParams)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::CellCycleParams>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::LigandParams)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::LigandParams)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::LigandParams>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::PolarizationAxisParams)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::PolarizationAxisParams)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::PolarizationAxisParams>)

#endif
