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

#ifndef _CUSTOMEVLDIVISIONFUNCTORS_H
#define _CUSTOMEVLDIVISIONFUNCTORS_H

namespace mg {

struct custom_evl_growth_division
  {
    d3*                 evlPosition;
    d3*                 evlNormal;
    d3*                 evlRadius;
    uint*               evlTimer;
    const double*       evlPressure;
    const uint          mPhaseLength;
    const double        evlGrowthThreshold;
    const double        evlLateralGrowthRatio;
    const double        evlRadiusLimit;
    double*             randomUniform;
    uint*               randomUniform_Counter;
    const uint          currentTimer;
    uint*               NumEVLCells;

    custom_evl_growth_division(
          d3*           _evlPosition,
          d3*           _evlNormal,
          d3*           _evlRadius,
          uint*         _evlTimer,
          double*       _evlPressure,
          uint          _mPhaseLength,
          double        _evlGrowthThreshold,
          double        _evlLateralGrowthRatio,
          double        _evlRadiusLimit,
          double*       _randomUniform,
          uint*         _randomUniformCounter,
          uint          _currentTimer,
          uint*         _NumEVLCells
        )
         :
            evlPosition(_evlPosition), 
            evlNormal(_evlNormal),
            evlRadius(_evlRadius),
            evlTimer(_evlTimer),
            evlPressure(_evlPressure),
            mPhaseLength(_mPhaseLength),
            evlGrowthThreshold(_evlGrowthThreshold),
            evlLateralGrowthRatio(_evlLateralGrowthRatio),
            evlRadiusLimit(_evlRadiusLimit),
            randomUniform(_randomUniform),
            randomUniform_Counter(_randomUniformCounter),
            currentTimer(_currentTimer),
            NumEVLCells(_NumEVLCells)
            {}

    __device__
    void operator()(const int& idx){

      // "State"
      uint mphase = 0;
      if( evlTimer[idx] < mPhaseLength ){
        mphase = 1;
      }

    	// Growth
      double radius_l = evlRadius[idx].x;
      double pressure = evlPressure[idx];

      if( pressure > evlGrowthThreshold && mphase == 0 ){
        radius_l *= evlLateralGrowthRatio;
      }
      else if(pressure < 0 && mphase == 0 ){
        radius_l /= evlLateralGrowthRatio;
      }

      // Division
      uint divide = 0;
      if( radius_l > evlRadiusLimit && mphase == 0 ){
        // here we prevent that all evls divide at the same time step if tension is too high
        // by allowing the division randomly
        if ( randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] < .5 ){
          divide = 1;
        }
      }
      
      // if(idx == 0){
      //   printf("Evl %d radiusl %lf pressure %lf threshold %lf\n", idx, radius_l, pressure, evlGrowthThreshold);
      // }

      if(divide == 1){

        radius_l *= 0.70710678118;    // surface divided by 2, radius divided by 2^(1/2)

        uint sisID = mgAtomicAddOne(&NumEVLCells[0]);

        // printf("EVL %d divides, sister %d\n", idx, sisID);

        evlRadius[sisID].x = radius_l;

        // Random division axis
        d3 normal = evlNormal[idx];
        d3 normal2 = d3(-normal.y,normal.x,.0);
        normal2 /= length(normal2);
      
        d3 normal3 = cross(normal, normal2);

        double alpha = 2.0 * PI * randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0]) ];
        
        d3 ran = cos(alpha) * normal2 + sin(alpha) * normal3;

        double coeff = .8;

        d3 pos = evlPosition[idx];

        evlPosition[idx]    = pos + coeff * radius_l * ran;
        evlPosition[sisID]  = pos - coeff * radius_l * ran;

        evlNormal[sisID] = normal;

        evlTimer[idx] = currentTimer;
        evlTimer[sisID] = currentTimer;

      }

      evlRadius[idx].x = radius_l;
    }
  };

}

#endif