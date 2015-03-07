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

#ifndef DISCONWIDGET_H
#define DISCONWIDGET_H

#include <QWidget>
#include <QSlider>

// Project includes:
#include "thrust_objects.hpp"
// #include "metaparam.hpp"
#include "param.hpp"
// #include "cells_producer.hpp"

#include "glwidget.hpp"



namespace mg{

  class DisplayControlWidget : public QWidget
  {
    Q_OBJECT

    public:
      explicit DisplayControlWidget(GLWidget *_glw, QWidget *parent = 0, Param<HOST> * _p = 0);
      ~DisplayControlWidget();

      QSize minimumSizeHint() const;
      QSize sizeHint()        const;

    public slots:

    signals:

    protected:

      

    private:
      
      QSlider * sliderSlicerXmax;
      QSlider * sliderSlicerYmax;
      QSlider * sliderSlicerZmax;
      QSlider * sliderSlicerXmin;
      QSlider * sliderSlicerYmin;
      QSlider * sliderSlicerZmin;

      GLWidget *glwidget;

      Param<HOST> * param;
  };

}

#endif

