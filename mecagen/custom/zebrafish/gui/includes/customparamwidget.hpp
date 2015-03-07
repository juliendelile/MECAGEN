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

      QSlider * fprotr_Slider;
      QSlider * evlstiff_Slider;
      QSlider * evlGratio_Slider;
      QSlider * icYolkAttr_Slider;
      QSlider * evlSizeLimit_Slider;
      QSlider * margRes_Slider;
      QSlider * evlGthres_Slider;
      QSlider * icAttr_Slider;
      QSlider * icRep_Slider;

      Window      *window;

      public slots:
     
      void setFProtrusion(int val);
      void setEvlStiffness(int val);
      void setEvlGrowthRatio(int val);
      void setICYolkAttr(int val);
      void setEvlSizeLimit(int val);
      void setMarginResistance(int val);
      void setEvlGrowthThreshold(int val);
      void setICAttr(int val);
      void setICRep(int val);
  };

}

#endif

