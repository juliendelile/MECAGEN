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

#ifndef _CUSTOMDIFFUSIONFUNCTORS_H
#define _CUSTOMDIFFUSIONFUNCTORS_H

namespace mg {

struct custom_yolk_evl_diffusion
  {
    const uint*     cellsYolkNeighbNum;
    const uint*     cellsEvlNeighbNum;
    const uint      yolkLigandId;
    const uint      evlLigandId;
    const double    yolkLigandUpdate;
    const double    evlLigandUpdate;
    double*         cellLigand;
    double*         cellLigandUpdate;
    
    custom_yolk_evl_diffusion(
          uint*       _cellsYolkNeighbNum,
          uint*       _cellsEvlNeighbNum,
          uint        _yolkLigandId,
          uint        _evlLigandId,
          double      _yolkLigandUpdate,
          double      _evlLigandUpdate,
          double*     _cellLigand,
          double*     _cellLigandUpdate
        )
         :
            cellsYolkNeighbNum(_cellsYolkNeighbNum),
            cellsEvlNeighbNum(_cellsEvlNeighbNum),
            yolkLigandId(_yolkLigandId),
            evlLigandId(_evlLigandId),
            yolkLigandUpdate(_yolkLigandUpdate),
            evlLigandUpdate(_evlLigandUpdate),
            cellLigand(_cellLigand),
            cellLigandUpdate(_cellLigandUpdate)
            {}

    __device__
    void operator()(const int& idx){

      
      if(cellsYolkNeighbNum[idx] > 0){
        // Yolk as source
        if(yolkLigandUpdate > .0){
          cellLigandUpdate[NUMLIGmax*idx + yolkLigandId] += yolkLigandUpdate;
        }
        // Yolk as sink
        cellLigand[NUMLIGmax*idx + evlLigandId] = .0;
        cellLigandUpdate[NUMLIGmax*idx + evlLigandId] = .0;
      }

      
      if(cellsEvlNeighbNum[idx] > 0){
        // EVL as source
        if(evlLigandUpdate > .0){
          cellLigandUpdate[NUMLIGmax*idx + evlLigandId] += evlLigandUpdate;
        }
        // EVL as sink
        cellLigand[NUMLIGmax*idx + yolkLigandId] = .0;
        cellLigandUpdate[NUMLIGmax*idx + yolkLigandId] = .0;
      }

    }
  };

}

#endif