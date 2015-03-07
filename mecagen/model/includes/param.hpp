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

#ifndef _PARAM_H
#define _PARAM_H

// Project:
#include "param_objects.hpp"

#include "param_grn.hpp"

#include "thrust_objects.hpp"     // Backend<>

// -- Collection related
#include <vector>

#include "custom.hpp"

namespace mg {

  /** Param object.
   * This object must be deeply copiable, and follow the Rule of the Great Three. Here, we use thrust vectors
   * as members so it is indeed the case. However, in the Host-Device mode of the framework, an additional method
   * must be implemented to allow the copy from the backend to the device backend ("copy" method).
   */
  template<int T>
    class Param{

      public:

        /** Param class constructor.
         * The parameter values are specified here.
         */
        Param():
          globalDamping(1),
          randomGaussian(400000),
          randomUniform(400000),
          deltaTime(1),
          numLigands(1),
          ligandParams(NUMLIGmax),
          cellCycleParams(1),
          numPolarizationAxes(1),
          polarizationAxisParams(NUMAXESmax),

          customParam(1),

          numProteinNodes(1),
          numProteins(1),
          numPPInteractions(1),
          numReceptors(1),
          numTransReceptors(1),
          numSecretors(1),
          numGenes(1),
          numPolarizationNodes(1),
          numEpiPolarizationNodes(1),
          numAdhesionNodes(1),
          proteinNodes(NUMPROTEINNODEmax),
          proteins(NUMPROTEINmax),
          genes(NUMGENEmax),
          receptors(NUMRECEPTORmax),
          transReceptors(NUMTRANSRECEPTORmax),
          secretors(NUMSECRETORmax),
          ppInteractions(NUMPPINTERACTIONmax),
          cellTypeNodes(2),
          polarizationNodes(NUMPOLARIZATIONNODEmax),
          epiPolarizationNodes(NUMPOLARIZATIONNODEmax),
          forcePolarizationNode(1),
          mechanoSensorNode(1),
          protrusionNode(1),
          bipolarityNode(1),
          adhesionNodes(NUMADHESIONNODEmax),
          mechaParams(1)
      { }

        /** Param class destructor. */
        ~Param() throw () {
        }

        public:

        /** Is used to store the global damping coefficent. */
        typename Backend<T>::vecDouble                        globalDamping;

        /** Is used to store a large number of random number following a gaussian distribution. */
        typename Backend<T>::vecDouble                        randomGaussian;

        /** Is used to store a large number of random number following a uniform distribution. */
        typename Backend<T>::vecDouble                        randomUniform;

        /** Is used to store the number of ligands. */
        typename Backend<T>::vecUint                          numLigands;

        /** Is used to store the set of parameters for each ligand. */
        typename BackendParam<T>::vecLigandParams             ligandParams;

        /** Is used to store the parameters of the cell cycle rule. */
        typename BackendParam<T>::vecCellCycleParams          cellCycleParams;

        /** Is used to store the number of candidate polarization axes. */
        typename Backend<T>::vecUint                          numPolarizationAxes;

        /** Is used to store the parameters of each candidate polarization axis. */
        typename BackendParam<T>::vecPolarizationAxisParams   polarizationAxisParams;

        /** Is used to store the timelength of a simulation step, in seconds.*/
        typename Backend<T>::vecDouble                        deltaTime;

        /** Is used to store the number of spatiotemporal protein synthesis nodes. */
        typename Backend<T>::vecUint                          numProteinNodes;

        /** Is used to store the number of proteins. */
        typename Backend<T>::vecUint                          numProteins;

        /** Is used to store the number of protein-protein interaction nodes. */
        typename Backend<T>::vecUint                          numPPInteractions;

        /** Is used to store the number of receptor nodes. */
        typename Backend<T>::vecUint                          numReceptors;

        /** Is used to store the number of trans-receptor nodes. */
        typename Backend<T>::vecUint                          numTransReceptors;

        /** Is used to store the number of secretor nodes. */
        typename Backend<T>::vecUint                          numSecretors;

        /** Is used to store the number of genes. */
        typename Backend<T>::vecUint                          numGenes;

        /** Is used to store the number of polarization nodes. */
        typename Backend<T>::vecUint                          numPolarizationNodes;

        /** Is used to store the number of polarization nodes used for epithelialization. */
        typename Backend<T>::vecUint                          numEpiPolarizationNodes;

        /** Is used to store the number of adhesion nodes. */
        typename Backend<T>::vecUint                          numAdhesionNodes;

        /** Is used to store the parameters of each spatiotemporal protein synthesis node. */
        typename BackendGrnParam<T>::vecProteinNode           proteinNodes;

        /** Is used to store the parameters of each protein. */
        typename BackendGrnParam<T>::vecProtein               proteins;

        /** Is used to store the parameters of each gene. */
        typename BackendGrnParam<T>::vecGene                  genes;

        /** Is used to store the parameters of each receptor node. */
        typename BackendGrnParam<T>::vecReceptor              receptors;

        /** Is used to store the parameters of each trans-receptor node. */
        typename BackendGrnParam<T>::vecReceptor              transReceptors;

        /** Is used to store the parameters of each secretor node. */
        typename BackendGrnParam<T>::vecSecretor              secretors;

        /** Is used to store the parameters of each protein-protein interaction node. */
        typename BackendGrnParam<T>::vecPPInteraction         ppInteractions;

        /** Is used to store the regulatory function of each archetype node. */
        typename BackendGrnParam<T>::vecRegulatoryElement     cellTypeNodes;

        /** Is used to store the parameters of each polarization nodes. */
        typename BackendGrnParam<T>::vecPolarizationNode      polarizationNodes;

        /** Is used to store the parameters of the force polarization node. */
        typename BackendGrnParam<T>::vecForcePolarizationNode forcePolarizationNode;

        /** Is used to store the parameters of the mechanotransduction node. */
        typename BackendGrnParam<T>::vecMechanoSensorNode     mechanoSensorNode;

        /** Is used to store the ids of each polarization node involved in epithelialization. */
        typename Backend<T>::vecUint                          epiPolarizationNodes;

        /** Is used to store the parameters of the protrusion nodes. */
        typename BackendGrnParam<T>::vecProtrusionNode        protrusionNode;

        /** Is used to store the parameters of the bipolarity node. */
        typename BackendGrnParam<T>::vecBipolarityNode        bipolarityNode;

        /** Is used to store the parameters of each adhesion node. */
        typename BackendGrnParam<T>::vecAdhesionNode          adhesionNodes;

        /** Is used to store the biomechanical parameters. */
        typename BackendGrnParam<T>::vecMechaParams           mechaParams;

        /** Is used to store the custom parameter object. */
        typename BackendCustom<T>::vecCustomParam             customParam;
    };
} // End namespace

#endif
