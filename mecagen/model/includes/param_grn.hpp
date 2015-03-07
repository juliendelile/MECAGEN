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

#ifndef _PARAMGRN_H
#define _PARAMGRN_H

#include "define.hpp"
#include "param_objects.hpp"

namespace mg {

  struct MechaParams{
    double maximumDistanceCoefficient[9];
    double surfaceScaling[9];
    double equilibriumDistanceCoefficient[9];
    double repulsionCoefficient[9];
    double planarRigidityCoefficient[9];

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(maximumDistanceCoefficient)
            & BOOST_SERIALIZATION_NVP(surfaceScaling)
            & BOOST_SERIALIZATION_NVP(equilibriumDistanceCoefficient)
            & BOOST_SERIALIZATION_NVP(repulsionCoefficient)
            & BOOST_SERIALIZATION_NVP(planarRigidityCoefficient)
          ;
      }
  };

  struct Protein{
    double kappa;
    char name[300];
    // double max;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(kappa)
            & BOOST_SERIALIZATION_NVP(name)
          ;
      }
  };

  struct RegulatoryElement{
    uint numInputProtein;
    uint inputProteinID[4];
    double inputThreshold[4];
    uint inputType[4];    //0 inhibitrice, 1 excitatrice
    uint logicalFunction; //0 and, 1 or

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(numInputProtein)
            & BOOST_SERIALIZATION_NVP(inputProteinID)
            & BOOST_SERIALIZATION_NVP(inputThreshold)
            & BOOST_SERIALIZATION_NVP(inputType)
            & BOOST_SERIALIZATION_NVP(logicalFunction)
          ;
      }
  };

  struct PolarizationNode{

    RegulatoryElement regEl;
    uint axisID;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(regEl)
            & BOOST_SERIALIZATION_NVP(axisID)
          ;
      }
  };

  struct ForcePolarizationNode{

    RegulatoryElement regEl;
    double force_threshold;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(regEl)
            & BOOST_SERIALIZATION_NVP(force_threshold)
          ;
      }
  };

  struct MechanoSensorNode{

    RegulatoryElement regEl;
    double force_threshold;
    double xi;
    uint outputProteinID;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(regEl)
            & BOOST_SERIALIZATION_NVP(force_threshold)
            & BOOST_SERIALIZATION_NVP(xi)
            & BOOST_SERIALIZATION_NVP(outputProteinID)
          ;
      }
  };

  struct AdhesionNode{

    uint              proteinID;
    uint              mode; // 0 min, 1 average, 2 ...
    double            k_adh;
    double            param1;
    double            param2;
    double            param3;
    double            param4;
    
    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(proteinID)
            & BOOST_SERIALIZATION_NVP(mode)
            & BOOST_SERIALIZATION_NVP(k_adh)
            & BOOST_SERIALIZATION_NVP(param1)
            & BOOST_SERIALIZATION_NVP(param2)
            & BOOST_SERIALIZATION_NVP(param3)
            & BOOST_SERIALIZATION_NVP(param4)
          ;
      }
  };

  struct BipolarityNode{

    RegulatoryElement     regEl;
    
    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(regEl)
          ;
      }
  };

  struct ProtrusionNode{

    RegulatoryElement regEl;
    double  force;
    uint    adhesionID;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(regEl)
            & BOOST_SERIALIZATION_NVP(force)
            & BOOST_SERIALIZATION_NVP(adhesionID)
          ;
      }
  };

  struct Gene{
    RegulatoryElement regEl;
    uint outputProteinID;
    double beta;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  
            & BOOST_SERIALIZATION_NVP(regEl)
            & BOOST_SERIALIZATION_NVP(outputProteinID)
            & BOOST_SERIALIZATION_NVP(beta)
          ;
      }
  };

  struct Receptor{

    double tau;
    uint receptorProtID;
    uint ligID;
    uint outputProtID;
    double x_receptorProt;
    double x_lig;
    uint alpha_lig;
    uint alpha_receptorProt;
    uint alpha_outputProt; 

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(tau)
            & BOOST_SERIALIZATION_NVP(receptorProtID)
            & BOOST_SERIALIZATION_NVP(ligID)
            & BOOST_SERIALIZATION_NVP(outputProtID)
            & BOOST_SERIALIZATION_NVP(x_receptorProt)
            & BOOST_SERIALIZATION_NVP(x_lig)
            & BOOST_SERIALIZATION_NVP(alpha_lig)
            & BOOST_SERIALIZATION_NVP(alpha_receptorProt)
            & BOOST_SERIALIZATION_NVP(alpha_outputProt)
          ;
      }
      
  };

  struct Secretor{
    uint outputLigandID;
    uint inputProteinID;
    double sigma;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(outputLigandID)
            & BOOST_SERIALIZATION_NVP(inputProteinID)
            & BOOST_SERIALIZATION_NVP(sigma)
          ;
      }
  };

  struct PPInteraction{
    uint numReactant;
    uint reactantID[4];
    uint x[4];
    uint alpha[4];
    uint outputProteinID;
    uint outputProteinAlpha;
    double k;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(numReactant)
            & BOOST_SERIALIZATION_NVP(reactantID)
            & BOOST_SERIALIZATION_NVP(x)
            & BOOST_SERIALIZATION_NVP(alpha)
            & BOOST_SERIALIZATION_NVP(outputProteinID)
            & BOOST_SERIALIZATION_NVP(outputProteinAlpha)
            & BOOST_SERIALIZATION_NVP(k)
          ;
      }
  };

  struct ProteinNode{
    d3 Xmin;
    d3 Xmax;
    uint tmin;
    uint tmax;
    uint outputProteinID;
    double quantity;

    private:
    friend class boost::serialization::access;

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){

        ar  & BOOST_SERIALIZATION_NVP(Xmin)
            & BOOST_SERIALIZATION_NVP(Xmax)
            & BOOST_SERIALIZATION_NVP(tmin)
            & BOOST_SERIALIZATION_NVP(tmax)
            & BOOST_SERIALIZATION_NVP(outputProteinID)
            & BOOST_SERIALIZATION_NVP(quantity)
          ;
      }
  };
  
  /** Backend template specialization mechanism for Param structure.
   * According to the backend indicator, corresponding types of thrust vectors are defined with the same 
   * denomination
   */
  template<int integer>
    struct BackendGrnParam;

  /** Device backend types specialization. */
  template<>
    struct BackendGrnParam<DEVICE>
    {
      typedef typename thrust::device_vector<ProteinNode>             vecProteinNode;
      typedef typename thrust::device_vector<Protein>                 vecProtein;
      typedef typename thrust::device_vector<Gene>                    vecGene;
      typedef typename thrust::device_vector<Receptor>                vecReceptor;
      typedef typename thrust::device_vector<Secretor>                vecSecretor;
      typedef typename thrust::device_vector<PPInteraction>           vecPPInteraction;
      typedef typename thrust::device_vector<PolarizationNode>        vecPolarizationNode;
      typedef typename thrust::device_vector<ForcePolarizationNode>   vecForcePolarizationNode;
      typedef typename thrust::device_vector<MechanoSensorNode>       vecMechanoSensorNode;
      typedef typename thrust::device_vector<RegulatoryElement>       vecRegulatoryElement;
      typedef typename thrust::device_vector<AdhesionNode>            vecAdhesionNode;
      typedef typename thrust::device_vector<BipolarityNode>          vecBipolarityNode;
      typedef typename thrust::device_vector<ProtrusionNode>          vecProtrusionNode;
      typedef typename thrust::device_vector<MechaParams>             vecMechaParams;
    };

  /** Host backend types specialization. */
  template<>
    struct BackendGrnParam<HOST>
    {
      typedef typename thrust::host_vector<ProteinNode>               vecProteinNode;
      typedef typename thrust::host_vector<Protein>                   vecProtein;
      typedef typename thrust::host_vector<Gene>                      vecGene;
      typedef typename thrust::host_vector<Receptor>                  vecReceptor;
      typedef typename thrust::host_vector<Secretor>                  vecSecretor;
      typedef typename thrust::host_vector<PPInteraction>             vecPPInteraction;
      typedef typename thrust::host_vector<PolarizationNode>          vecPolarizationNode;
      typedef typename thrust::host_vector<ForcePolarizationNode>     vecForcePolarizationNode;
      typedef typename thrust::host_vector<MechanoSensorNode>         vecMechanoSensorNode;
      typedef typename thrust::host_vector<RegulatoryElement>         vecRegulatoryElement;
      typedef typename thrust::host_vector<AdhesionNode>              vecAdhesionNode;
      typedef typename thrust::host_vector<BipolarityNode>            vecBipolarityNode;
      typedef typename thrust::host_vector<ProtrusionNode>            vecProtrusionNode;
      typedef typename thrust::host_vector<MechaParams>               vecMechaParams;
    };

} // End namespace

BOOST_SAVE_THRUST_HOST_VECTOR(mg::ProteinNode)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::ProteinNode)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::ProteinNode>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::Protein)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::Protein)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::Protein>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::Gene)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::Gene)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::Gene>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::Receptor)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::Receptor)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::Receptor>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::Secretor)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::Secretor)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::Secretor>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::PPInteraction)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::PPInteraction)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::PPInteraction>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::PolarizationNode)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::PolarizationNode)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::PolarizationNode>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::ForcePolarizationNode)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::ForcePolarizationNode)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::ForcePolarizationNode>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::MechanoSensorNode)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::MechanoSensorNode)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::MechanoSensorNode>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::RegulatoryElement)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::RegulatoryElement)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::RegulatoryElement>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::AdhesionNode)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::AdhesionNode)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::AdhesionNode>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::BipolarityNode)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::BipolarityNode)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::BipolarityNode>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::ProtrusionNode)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::ProtrusionNode)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::ProtrusionNode>)

BOOST_SAVE_THRUST_HOST_VECTOR(mg::MechaParams)
BOOST_LOAD_THRUST_HOST_VECTOR(mg::MechaParams)
BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::MechaParams>)

// BOOST_SAVE_THRUST_HOST_VECTOR(mg::)
// BOOST_LOAD_THRUST_HOST_VECTOR(mg::)
// BOOST_SERIALIZATION_SPLIT_FREE(thrust::host_vector<mg::>)

#endif
