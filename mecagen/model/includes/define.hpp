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

#define NUMAXESmax              7
#define NUMLIGmax               6

#define NUMCLEAVAGEmax 4
#define NUMGENEmax 5
#define NUMRECEPTORmax 3
#define NUMSECRETORmax 3
#define NUMTRANSRECEPTORmax 3
#define NUMPPINTERACTIONmax 3
#define NUMPROTEINNODEmax 6
#define NUMPROTEINmax 11
#define NUMGENEINPUTSmax 4
#define NUMPOLARIZATIONNODEmax 7
#define NUMADHESIONNODEmax 7

#define NUMTIMESTEP             10000

#define PIov2                   1.57079633
#define PI3ov4                  2.35619449
#define PI2                     6.28318531
#define PI                      3.14159265

#define COS_PI_6                .86602540378
#define COS_PI_4                .70710678118
#define COS_PI_3                .5
#define COS_2PI_3               -.5
#define SIN_2PI_3               .86602540378

#define APF_2D                  0.90689968211  //(PI / (2*sqrtf(3)))
#define APF_2D_RAD              0.95231280686  //pow(APF_2D, 1.0/2.0)

#define PROTRUSION_COEFF        .75

#define SURF_A                  1.3697
#define SURF_L                  1.2414

#define NUMCELLmax              5000

#define NUMNEIGHBMETRICmax      200
#define NUMNEIGHBTOPOmax        40  
#define NUMFORCEmax             (NUMNEIGHBTOPOmax+1) //each neighbor may add a force + the force vector of the cell