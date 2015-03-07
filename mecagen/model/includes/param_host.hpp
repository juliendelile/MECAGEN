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

#ifndef _PARAMHOST_H
#define _PARAMHOST_H

// // Project:
#include "param.hpp"

// // Standard :
// #include <cstdlib>      //srand

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

// // -- I/O related
// #include <iostream>
#include <string>       //std::getline
#include <fstream>      //ifstream
#include <sstream>      //istringstream
#include "stdio.h"      //sprintf
// #include "time.h"       //time

// // -- Collection related
#include <map>

namespace mg {
  
  /** Param object.
   * This object must be deeply copiable, and follow the Rule of the Great Three. Here, we use thrust vectors
   * as members so it is indeed the case. However, in the Host-Device mode of the framework, an additional method
   * must be implemented to allow the copy from the backend to the device backend ("copy" method).
   */
    class Param_Host: public Param<HOST> {

      public:

        /** Param class constructor.
         * The parameter values are specified here.
         */
        Param_Host()

        {
        }

        /** Param class destructor. */
        ~Param_Host() throw () {
        }

        void load_RandomGaussian(long int s){

          boost::mt19937                                      seed(s);    // Mersenne Twister
          boost::normal_distribution<double>                  dist(0,1);   // Normal Distribution
          boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > gen(seed,dist);    // Variate generator
       
          int p[400000]={};   
          const int nrolls=40000;  // number of experiments
          const int nstars=10 * 20;    // maximum number of stars to distribute
          
          for (int i = 0; i < 400000; i++){
            double d = gen();
            randomGaussian[i] = d;
            // std::cout << d << std::endl;
            if ((d>=-2.0)&&(d<2.0)) ++p[int(10.0*(d + 2.0))];
          }     

          // for (int i=0; i<40; ++i) {
          //   std::cout << i << "-" << (i+1) << ": ";
          //   std::cout << std::string(p[i]*nstars/nrolls,'*') << std::endl;
          // } 

        }

        void load_RandomUniform(long int s){

          boost::mt19937                         seed(s);
          boost::random::uniform_real_distribution< >     dist(0.0,1.0);
          boost::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution < > > random(seed,dist);
           
          const int nrolls=40000;  // number of experiments
          const int nstars=10 * 20;    // maximum number of stars to distribute

          const int nintervals=10; // number of intervals
          int p[nintervals]={};

          for (int i = 0; i < 400000; i++){
              double d = random();
              randomUniform[i] = d;
              ++p[int(nintervals*d)];
          }     

          // std::cout << "uniform_real_distribution (0.0,1.0):" << std::endl;
          // std::cout << std::fixed; std::cout.precision(1);

          // for (int i=0; i<nintervals; ++i) {
          //   std::cout << float(i)/nintervals << "-" << float(i+1)/nintervals << ": ";
          //   std::cout << std::string(p[i]*nstars/nrolls,'*') << std::endl;
          // }
        }

        void display(){

          std::cout << "deltaTime           : " << deltaTime[0] <<std::endl; 
          std::cout << "globalDamping       : " << globalDamping[0] <<std::endl; 

          std::cout << "cellCycleParams " << std::endl;
          std::cout << "             mode  " << cellCycleParams[0].mode << std::endl;
          std::cout << "             param1  " << cellCycleParams[0].param1 << std::endl;
          std::cout << "             param2  " << cellCycleParams[0].param2 << std::endl;
          std::cout << "             param3  " << cellCycleParams[0].param3 << std::endl;
          std::cout << "             param4  " << cellCycleParams[0].param4 << std::endl;
          std::cout << "             volume_ratio  " << cellCycleParams[0].volume_ratio << std::endl;
          std::cout << "             mPhaseLength  " << cellCycleParams[0].mPhaseLength << std::endl;

          std::cout << "numPolarizationAxes : " << numPolarizationAxes[0] << std::endl;
          for(uint i=0; i<numPolarizationAxes[0]; i++){
            std::cout << " Axis " << i << std::endl;
            std::cout << "            idlig                   " << polarizationAxisParams[i].idlig << std::endl;
            std::cout << "            compMode                " << polarizationAxisParams[i].compMode << std::endl;
            std::cout << "            apicoBasalInEpithelium  " << polarizationAxisParams[i].apicoBasalInEpithelium << std::endl;
            std::cout << "            param1                  " << polarizationAxisParams[i].param1 << std::endl;
          }

          std::cout << "numLigands          : " << numLigands[0] << std::endl;
          for(uint i=0; i<numLigands[0]; i++){
            std::cout << " Ligand " << i << " : " << std::endl; 
            std::cout << "            name        " << ligandParams[i].name << std::endl;
            std::cout << "            diffusion   " << ligandParams[i].diffusion << std::endl;
            std::cout << "            chi         " << ligandParams[i].chi << std::endl;
          }

          std::cout << "numProteins         : " << numProteins[0]        << std::endl;
          for(uint i=0; i<numProteins[0]; i++){
            std::cout << " Protein " << i << " : " << std::endl; 
            std::cout << "            name  " << proteins[i].name << std::endl;
            std::cout << "            kappa " << proteins[i].kappa << std::endl;
          }

          std::cout << "numGenes            : " << numGenes[0] << std::endl;
          for(uint i=0; i<numGenes[0]; i++){
            std::cout << " Gene   " << i << " : "; 
            if(genes[i].regEl.numInputProtein == 0){std::cout << "no input" << std::endl;}
            else{
              std::cout << std::endl;
              std::cout << "            outputProteinID " << genes[i].outputProteinID << std::endl;
              std::cout << "            beta            " << genes[i].beta << std::endl;
              std::cout << "            numInputProtein " << genes[i].regEl.numInputProtein << std::endl;
              std::cout << "            inputProteinID  "; for(uint j=0; j<genes[i].regEl.numInputProtein;j++){std::cout << genes[i].regEl.inputProteinID[j] << " (" << proteins[genes[i].regEl.inputProteinID[j]].name << ") ";} std::cout << std::endl;
              std::cout << "            inputThreshold  "; for(uint j=0; j<genes[i].regEl.numInputProtein;j++){std::cout << genes[i].regEl.inputThreshold[j] << " ";} std::cout << std::endl;
              std::cout << "            inputType       "; for(uint j=0; j<genes[i].regEl.numInputProtein;j++){std::cout << genes[i].regEl.inputType[j] << " ";} std::cout << std::endl;
              std::cout << "            logicalFunction " << genes[i].regEl.logicalFunction << std::endl;
            }
          }
          
          std::cout << "numReceptors        : " << numReceptors[0]      << std::endl;
          for(uint i=0; i<numReceptors[0]; i++){
            std::cout << " Receptor " << i << " : " << std::endl; 
            std::cout << "            tau                 " << receptors[i].tau << std::endl;
            std::cout << "            receptorProtID      " << receptors[i].receptorProtID << std::endl;
            std::cout << "            ligID               " << receptors[i].ligID << std::endl;
            std::cout << "            outputProtID        " << receptors[i].outputProtID << std::endl;
            std::cout << "            x_receptorProt      " << receptors[i].x_receptorProt << std::endl;
            std::cout << "            x_lig               " << receptors[i].x_lig << std::endl;
            std::cout << "            alpha_lig           " << receptors[i].alpha_lig << std::endl;
            std::cout << "            alpha_receptorProt  " << receptors[i].alpha_receptorProt << std::endl;
            std::cout << "            alpha_outputProt    " << receptors[i].alpha_outputProt << std::endl;
          }
          
          std::cout << "numTransReceptors   : " << numTransReceptors[0]      << std::endl;
          for(uint i=0; i<numTransReceptors[0]; i++){
            std::cout << " TransReceptor " << i << " : " << std::endl; 
            std::cout << "            tau                 " << transReceptors[i].tau << std::endl;
            std::cout << "            receptorProtID      " << transReceptors[i].receptorProtID << std::endl;
            std::cout << "            ligID               " << transReceptors[i].ligID << std::endl;
            std::cout << "            outputProtID        " << transReceptors[i].outputProtID << std::endl;
            std::cout << "            x_receptorProt      " << transReceptors[i].x_receptorProt << std::endl;
            std::cout << "            x_lig               " << transReceptors[i].x_lig << std::endl;
            std::cout << "            alpha_lig           " << transReceptors[i].alpha_lig << std::endl;
            std::cout << "            alpha_receptorProt  " << transReceptors[i].alpha_receptorProt << std::endl;
            std::cout << "            alpha_outputProt    " << transReceptors[i].alpha_outputProt << std::endl;
          }
          
          std::cout << "numSecretors        : " << numSecretors[0]      << std::endl;
          for(uint i=0; i<numSecretors[0]; i++){
            std::cout << " Secretor " << i << " : " << std::endl; 
            std::cout << "            outputLigandID  " << secretors[i].outputLigandID << std::endl;
            std::cout << "            inputProteinID  " << secretors[i].inputProteinID << std::endl;
            std::cout << "            sigma           " << secretors[i].sigma << std::endl;
          }

          std::cout << "numProteinNodes     : " << numProteinNodes[0]   << std::endl;
          for(uint i=0; i<numProteinNodes[0]; i++){
            std::cout << " ProteinNode " << i << " : " << std::endl; 
            std::cout << "            outputProteinID " << proteinNodes[i].outputProteinID << std::endl;
            std::cout << "            quantity        " << proteinNodes[i].quantity << std::endl;
            std::cout << "            tmin            " << proteinNodes[i].tmin << std::endl;
            std::cout << "            tmax            " << proteinNodes[i].tmax << std::endl;
            std::cout << "            Xmin            " << proteinNodes[i].Xmin.x << " " << proteinNodes[i].Xmin.y << " " << proteinNodes[i].Xmin.z << std::endl;
            std::cout << "            Xmax            " << proteinNodes[i].Xmax.x << " " << proteinNodes[i].Xmax.y << " " << proteinNodes[i].Xmax.z << std::endl;
          }

          std::cout << "numPPInteractions   : " << numPPInteractions[0] << std::endl;
          for(uint i=0; i<numPPInteractions[0]; i++){
            std::cout << " PPInteraction " << i << " : " << std::endl; 
            std::cout << "            numReactant         " << ppInteractions[i].numReactant << std::endl;
            std::cout << "            reactantID          "; for(uint j=0; j<ppInteractions[i].numReactant;j++){std::cout << ppInteractions[i].reactantID[j] << " ";} std::cout << std::endl;
            std::cout << "            x                   "; for(uint j=0; j<ppInteractions[i].numReactant;j++){std::cout << ppInteractions[i].x[j] << " ";} std::cout << std::endl;
            std::cout << "            alpha               "; for(uint j=0; j<ppInteractions[i].numReactant;j++){std::cout << ppInteractions[i].alpha[j] << " ";} std::cout << std::endl;
            std::cout << "            outputProteinID     " << ppInteractions[i].outputProteinID << std::endl;
            std::cout << "            outputProteinAlpha  " << ppInteractions[i].outputProteinAlpha << std::endl;
            std::cout << "            k                   " << ppInteractions[i].k << std::endl;
          }

          std::cout << "numAdhesionNodes   : " << numAdhesionNodes[0] << std::endl;
          for(uint i=0; i<numAdhesionNodes[0]; i++){
            std::cout << " AdhesionNode " << i << " : " << std::endl; 
            std::cout << "            mode                " << adhesionNodes[i].mode << std::endl;
            std::cout << "            k_adh               " << adhesionNodes[i].k_adh << std::endl;
            std::cout << "            proteinID           " << adhesionNodes[i].proteinID << std::endl;
            std::cout << "            params              " << adhesionNodes[i].param1 << " " << adhesionNodes[i].param2 << " " << adhesionNodes[i].param3 << " " << adhesionNodes[i].param4 << std::endl;
          }

          std::cout << "Mesenchymal Type Node  : ";
          if(cellTypeNodes[0].numInputProtein == 0){std::cout << "no input" << std::endl;}
          else{
            std::cout << std::endl; 
            std::cout << "            numInputProtein " << cellTypeNodes[0].numInputProtein << std::endl;
            std::cout << "            inputProteinID  "; for(uint j=0; j<cellTypeNodes[0].numInputProtein;j++){std::cout << cellTypeNodes[0].inputProteinID[j] << " ";} std::cout << std::endl;
            std::cout << "            inputThreshold  "; for(uint j=0; j<cellTypeNodes[0].numInputProtein;j++){std::cout << cellTypeNodes[0].inputThreshold[j] << " ";} std::cout << std::endl;
            std::cout << "            inputType       "; for(uint j=0; j<cellTypeNodes[0].numInputProtein;j++){std::cout << cellTypeNodes[0].inputType[j] << " ";} std::cout << std::endl;
            std::cout << "            logicalFunction " << cellTypeNodes[0].logicalFunction << std::endl;
          }

          std::cout << "Epithelial Type Node  : "; 
          if(cellTypeNodes[1].numInputProtein == 0){std::cout << "no input" << std::endl;}
          else{
            std::cout << std::endl; 
            std::cout << "            numInputProtein " << cellTypeNodes[1].numInputProtein << std::endl;
            std::cout << "            inputProteinID  "; for(uint j=0; j<cellTypeNodes[1].numInputProtein;j++){std::cout << cellTypeNodes[1].inputProteinID[j] << " ";} std::cout << std::endl;
            std::cout << "            inputThreshold  "; for(uint j=0; j<cellTypeNodes[1].numInputProtein;j++){std::cout << cellTypeNodes[1].inputThreshold[j] << " ";} std::cout << std::endl;
            std::cout << "            inputType       "; for(uint j=0; j<cellTypeNodes[1].numInputProtein;j++){std::cout << cellTypeNodes[1].inputType[j] << " ";} std::cout << std::endl;
            std::cout << "            logicalFunction " << cellTypeNodes[1].logicalFunction << std::endl;
          }
          
          std::cout << "numPolarizationNodes : "  << numPolarizationNodes[0] <<std::endl;
          for(uint i=0; i<numPolarizationNodes[0]; i++){
            std::cout << "  Polarization Node " << i << " : ";
            if(polarizationNodes[i].regEl.numInputProtein == 0){std::cout << "no input" << std::endl;}
            else{
              std::cout << std::endl;
              std::cout << "            axisID " << polarizationNodes[i].axisID << std::endl;
              std::cout << "            numInputProtein " << polarizationNodes[i].regEl.numInputProtein << std::endl;
              std::cout << "            inputProteinID  "; for(uint j=0; j<polarizationNodes[i].regEl.numInputProtein;j++){std::cout << polarizationNodes[i].regEl.inputProteinID[j] << " ";} std::cout << std::endl;
              std::cout << "            inputThreshold  "; for(uint j=0; j<polarizationNodes[i].regEl.numInputProtein;j++){std::cout << polarizationNodes[i].regEl.inputThreshold[j] << " ";} std::cout << std::endl;
              std::cout << "            inputType       "; for(uint j=0; j<polarizationNodes[i].regEl.numInputProtein;j++){std::cout << polarizationNodes[i].regEl.inputType[j] << " ";} std::cout << std::endl;
              std::cout << "            logicalFunction " << polarizationNodes[i].regEl.logicalFunction << std::endl;
            }
          }

          std::cout << "numEpiPolarizationNodes : " << numEpiPolarizationNodes[0] << std::endl;
          for(uint i=0; i < numEpiPolarizationNodes[0]; i++){
            std::cout << "            PolaNode index " << epiPolarizationNodes[i] << std::endl;
          }

          std::cout <<"Force Polarization Node : ";
          if(forcePolarizationNode[0].regEl.numInputProtein == 0){std::cout << "no input" << std::endl;}
          else{
            std::cout << std::endl;
            std::cout << "            force threshold " << forcePolarizationNode[0].force_threshold << std::endl;
            std::cout << "            numInputProtein " << forcePolarizationNode[0].regEl.numInputProtein << std::endl;
            std::cout << "            inputProteinID  "; for(uint j=0; j<forcePolarizationNode[0].regEl.numInputProtein;j++){std::cout << forcePolarizationNode[0].regEl.inputProteinID[j] << " ";} std::cout << std::endl;
            std::cout << "            inputThreshold  "; for(uint j=0; j<forcePolarizationNode[0].regEl.numInputProtein;j++){std::cout << forcePolarizationNode[0].regEl.inputThreshold[j] << " ";} std::cout << std::endl;
            std::cout << "            inputType       "; for(uint j=0; j<forcePolarizationNode[0].regEl.numInputProtein;j++){std::cout << forcePolarizationNode[0].regEl.inputType[j] << " ";} std::cout << std::endl;
            std::cout << "            logicalFunction " << forcePolarizationNode[0].regEl.logicalFunction << std::endl;
          }

          std::cout <<"MechanoSensor Node : ";
          if(mechanoSensorNode[0].regEl.numInputProtein == 0){std::cout << "no input" << std::endl;}
          else{
            std::cout << std::endl;
            std::cout << "            force threshold " << mechanoSensorNode[0].force_threshold << std::endl;
            std::cout << "            xi " << mechanoSensorNode[0].xi << std::endl;
            std::cout << "            outputProteinID " << mechanoSensorNode[0].outputProteinID << std::endl;
            std::cout << "            numInputProtein " << mechanoSensorNode[0].regEl.numInputProtein << std::endl;
            std::cout << "            inputProteinID  "; for(uint j=0; j<mechanoSensorNode[0].regEl.numInputProtein;j++){std::cout << mechanoSensorNode[0].regEl.inputProteinID[j] << " ";} std::cout << std::endl;
            std::cout << "            inputThreshold  "; for(uint j=0; j<mechanoSensorNode[0].regEl.numInputProtein;j++){std::cout << mechanoSensorNode[0].regEl.inputThreshold[j] << " ";} std::cout << std::endl;
            std::cout << "            inputType       "; for(uint j=0; j<mechanoSensorNode[0].regEl.numInputProtein;j++){std::cout << mechanoSensorNode[0].regEl.inputType[j] << " ";} std::cout << std::endl;
            std::cout << "            logicalFunction " << mechanoSensorNode[0].regEl.logicalFunction << std::endl;
          }

          std::cout <<"Protrusion Node : ";
          if(protrusionNode[0].regEl.numInputProtein == 0){std::cout << "no input" << std::endl;}
          else{
            std::cout << std::endl;
            std::cout << "            force " << protrusionNode[0].force << std::endl;
            std::cout << "            adhesionID " << protrusionNode[0].adhesionID << std::endl;
            std::cout << "            numInputProtein " << protrusionNode[0].regEl.numInputProtein << std::endl;
            std::cout << "            inputProteinID  "; for(uint j=0; j<protrusionNode[0].regEl.numInputProtein;j++){std::cout << protrusionNode[0].regEl.inputProteinID[j] << " ";} std::cout << std::endl;
            std::cout << "            inputThreshold  "; for(uint j=0; j<protrusionNode[0].regEl.numInputProtein;j++){std::cout << protrusionNode[0].regEl.inputThreshold[j] << " ";} std::cout << std::endl;
            std::cout << "            inputType       "; for(uint j=0; j<protrusionNode[0].regEl.numInputProtein;j++){std::cout << protrusionNode[0].regEl.inputType[j] << " ";} std::cout << std::endl;
            std::cout << "            logicalFunction " << protrusionNode[0].regEl.logicalFunction << std::endl;
          }

          std::cout <<"Bipolarity Node :";
          if(bipolarityNode[0].regEl.numInputProtein == 0){std::cout << "no input" << std::endl;}
          else{
            std::cout << std::endl;
            std::cout << "            numInputProtein " << bipolarityNode[0].regEl.numInputProtein << std::endl;
            std::cout << "            inputProteinID  "; for(uint j=0; j<bipolarityNode[0].regEl.numInputProtein;j++){std::cout << bipolarityNode[0].regEl.inputProteinID[j] << " ";} std::cout << std::endl;
            std::cout << "            inputThreshold  "; for(uint j=0; j<bipolarityNode[0].regEl.numInputProtein;j++){std::cout << bipolarityNode[0].regEl.inputThreshold[j] << " ";} std::cout << std::endl;
            std::cout << "            inputType       "; for(uint j=0; j<bipolarityNode[0].regEl.numInputProtein;j++){std::cout << bipolarityNode[0].regEl.inputType[j] << " ";} std::cout << std::endl;
            std::cout << "            logicalFunction " << bipolarityNode[0].regEl.logicalFunction << std::endl;
          }

          std::cout << "Mecha Parameters :" << std::endl;
          std::cout << " maximumDistanceCoefficient : "; for(int i=0; i<9; i++){std::cout << mechaParams[0].maximumDistanceCoefficient[i] << " ";} std::cout << std::endl;
          std::cout << " surfaceScaling : "; for(int i=0; i<9; i++){std::cout << mechaParams[0].surfaceScaling[i] << " ";} std::cout << std::endl;
          std::cout << " equilibriumDistanceCoefficient : "; for(int i=0; i<9; i++){std::cout << mechaParams[0].equilibriumDistanceCoefficient[i] << " ";} std::cout << std::endl;
          std::cout << " repulsionCoefficient : "; for(int i=0; i<9; i++){std::cout << mechaParams[0].repulsionCoefficient[i] << " ";} std::cout << std::endl;
          std::cout << " planarRigidityCoefficient : "; for(int i=0; i<9; i++){std::cout << mechaParams[0].planarRigidityCoefficient[i] << " ";} std::cout << std::endl;
        }

        /*** - BOOST SERIALIZATION - ***/
        private:
          friend class boost::serialization::access;


          template<class Archive>
            void serialize(Archive & ar, const unsigned int version){

              ar  
                  & BOOST_SERIALIZATION_NVP(globalDamping)
                  & BOOST_SERIALIZATION_NVP(deltaTime)
                  & BOOST_SERIALIZATION_NVP(numLigands)
                  & BOOST_SERIALIZATION_NVP(ligandParams)
                  & BOOST_SERIALIZATION_NVP(cellCycleParams)
                  & BOOST_SERIALIZATION_NVP(numPolarizationAxes)
                  & BOOST_SERIALIZATION_NVP(polarizationAxisParams)
                  & BOOST_SERIALIZATION_NVP(numProteinNodes)
                  & BOOST_SERIALIZATION_NVP(numProteins)
                  & BOOST_SERIALIZATION_NVP(numPPInteractions)
                  & BOOST_SERIALIZATION_NVP(numReceptors)
                  & BOOST_SERIALIZATION_NVP(numTransReceptors)
                  & BOOST_SERIALIZATION_NVP(numSecretors)
                  & BOOST_SERIALIZATION_NVP(numGenes)
                  & BOOST_SERIALIZATION_NVP(numPolarizationNodes)
                  & BOOST_SERIALIZATION_NVP(numEpiPolarizationNodes)
                  & BOOST_SERIALIZATION_NVP(numAdhesionNodes)
                  & BOOST_SERIALIZATION_NVP(proteinNodes)
                  & BOOST_SERIALIZATION_NVP(proteins)
                  & BOOST_SERIALIZATION_NVP(genes)
                  & BOOST_SERIALIZATION_NVP(receptors)
                  & BOOST_SERIALIZATION_NVP(transReceptors)
                  & BOOST_SERIALIZATION_NVP(secretors)
                  & BOOST_SERIALIZATION_NVP(ppInteractions)
                  & BOOST_SERIALIZATION_NVP(cellTypeNodes)
                  & BOOST_SERIALIZATION_NVP(polarizationNodes)
                  & BOOST_SERIALIZATION_NVP(forcePolarizationNode)
                  & BOOST_SERIALIZATION_NVP(mechanoSensorNode)
                  & BOOST_SERIALIZATION_NVP(epiPolarizationNodes)
                  & BOOST_SERIALIZATION_NVP(protrusionNode)
                  & BOOST_SERIALIZATION_NVP(bipolarityNode)
                  & BOOST_SERIALIZATION_NVP(adhesionNodes)
                  & BOOST_SERIALIZATION_NVP(mechaParams)
                  & BOOST_SERIALIZATION_NVP(customParam)
                  ;
            }

    };
} // End namespace

#endif
