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

#ifndef _PARAMDEVICE_H
#define _PARAMDEVICE_H

#include "param.hpp"

namespace mg {

  /** Param object.
   * This object must be deeply copiable, and follow the Rule of the Great Three. Here, we use thrust vectors
   * as members so it is indeed the case. However, in the Host-Device mode of the framework, an additional method
   * must be implemented to allow the copy from the backend to the device backend ("copy" method).
   */
    class Param_Device: public Param<DEVICE> {

      public:

        /** Param class constructor.
         * The parameter values are specified here.
         */
        Param_Device()
        {}

        /** Param class destructor. */
        ~Param_Device() throw () {
        }

        /** Required by the ISF framework in the Host-Device mode. Each time that the simulation loop is started, 
         * a copy from the host backend to the device backend is called internally via this method. 
         */
         
        Param_Device& copyFromHost(Param_Host & other){
            globalDamping = other.globalDamping;
            randomGaussian = other.randomGaussian;
            randomUniform = other.randomUniform;
            deltaTime = other.deltaTime;
            numLigands = other.numLigands;
            ligandParams = other.ligandParams;
            cellCycleParams = other.cellCycleParams;
            numPolarizationAxes = other.numPolarizationAxes;
            polarizationAxisParams = other.polarizationAxisParams;
            
            customParam = other.customParam;

            numProteinNodes = other.numProteinNodes;
            numProteins = other.numProteins;
            numPPInteractions = other.numPPInteractions;
            numReceptors = other.numReceptors;
            numTransReceptors = other.numTransReceptors;
            numSecretors = other.numSecretors;
            numGenes = other.numGenes;
            numPolarizationNodes = other.numPolarizationNodes;
            numEpiPolarizationNodes = other.numEpiPolarizationNodes;
            numAdhesionNodes = other.numAdhesionNodes;
            proteinNodes = other.proteinNodes;
            proteins = other.proteins;
            genes = other.genes;
            receptors = other.receptors;
            transReceptors = other.transReceptors;
            secretors = other.secretors;
            ppInteractions = other.ppInteractions;
            cellTypeNodes = other.cellTypeNodes;
            polarizationNodes = other.polarizationNodes;
            forcePolarizationNode = other.forcePolarizationNode;
            mechanoSensorNode = other.mechanoSensorNode;
            epiPolarizationNodes = other.epiPolarizationNodes;
            protrusionNode = other.protrusionNode;
            bipolarityNode = other.bipolarityNode;
            adhesionNodes = other.adhesionNodes;
            mechaParams = other.mechaParams;

            return *this;
          } 


    };
} // End namespace

#endif
