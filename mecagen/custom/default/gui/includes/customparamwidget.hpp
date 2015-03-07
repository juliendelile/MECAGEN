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

#ifndef CUSTOMPARAMWIDGETWIDGET_H
#define CUSTOMPARAMWIDGETWIDGET_H


// Project includes:
#include "thrust_objects.hpp"
#include "param.hpp"

#include <QWidget>
#include <QSlider>


namespace mg{

  class Window;
  template<int T> class Param;

  class CustomParamWidget : public QWidget
  {
    Q_OBJECT

    public:
      explicit CustomParamWidget(Window *_win, Param<HOST> * _p, QWidget *parent = 0);
      ~CustomParamWidget();

      QSize minimumSizeHint() const;
      QSize sizeHint()        const;

      void initSliders();

    protected:


    private:

      Param<HOST> *     param;

      QSlider * w_adh_homotypic_Slider;
      QSlider * w_adh_heterotypic_Slider;
      QSlider * w_rep_Slider;
      QSlider * k_rig_Slider;
      QSlider * gab_coeff_Slider;
      QSlider * c_max_Slider;
      QSlider * damping_Slider;

      Window      *window;
  };

}

#endif

