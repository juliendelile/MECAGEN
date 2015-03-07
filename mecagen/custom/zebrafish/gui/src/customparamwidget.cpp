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

#include "customparamwidget.hpp"
#include "window.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>

namespace mg{

  //
  // Constructor/destructor:
  //

  CustomParamWidget::CustomParamWidget(Window *_win, Param<HOST> * _p, QWidget *parent): window(_win), param(_p)
  {

    setWindowFlags(Qt::WindowTitleHint | Qt::CustomizeWindowHint);

    fprotr_Slider                  = new QSlider(Qt::Vertical); fprotr_Slider->setRange(0,1000); fprotr_Slider->setSingleStep(1);
    evlstiff_Slider                = new QSlider(Qt::Vertical); evlstiff_Slider->setRange(0,100); evlstiff_Slider->setSingleStep(1);
    evlGratio_Slider               = new QSlider(Qt::Vertical); evlGratio_Slider->setRange(0,200); evlGratio_Slider->setSingleStep(1);
    icYolkAttr_Slider              = new QSlider(Qt::Vertical); icYolkAttr_Slider->setRange(0,1000); icYolkAttr_Slider->setSingleStep(1);
    evlSizeLimit_Slider            = new QSlider(Qt::Vertical); evlSizeLimit_Slider->setRange(0,1000); evlSizeLimit_Slider->setSingleStep(1);
    evlGthres_Slider               = new QSlider(Qt::Vertical); evlGthres_Slider->setRange(0,1000); evlGthres_Slider->setSingleStep(1);
    margRes_Slider                 = new QSlider(Qt::Vertical); margRes_Slider->setRange(0,1000); margRes_Slider->setSingleStep(1);
    icAttr_Slider                  = new QSlider(Qt::Vertical); icAttr_Slider->setRange(0,1000); icAttr_Slider->setSingleStep(1);
    icRep_Slider                   = new QSlider(Qt::Vertical); icRep_Slider->setRange(0,1000); icRep_Slider->setSingleStep(1);

    connect(fprotr_Slider,              SIGNAL(valueChanged(int)), this, SLOT(setFProtrusion(int)));
    connect(evlstiff_Slider,            SIGNAL(valueChanged(int)), this, SLOT(setEvlStiffness(int)));
    connect(evlGratio_Slider,           SIGNAL(valueChanged(int)), this, SLOT(setEvlGrowthRatio(int)));
    connect(icYolkAttr_Slider,          SIGNAL(valueChanged(int)), this, SLOT(setICYolkAttr(int)));
    connect(evlSizeLimit_Slider,        SIGNAL(valueChanged(int)), this, SLOT(setEvlSizeLimit(int)));
    connect(evlGthres_Slider,           SIGNAL(valueChanged(int)), this, SLOT(setEvlGrowthThreshold(int)));
    connect(margRes_Slider,             SIGNAL(valueChanged(int)), this, SLOT(setMarginResistance(int)));
    connect(icAttr_Slider,              SIGNAL(valueChanged(int)), this, SLOT(setICAttr(int)));
    connect(icRep_Slider,               SIGNAL(valueChanged(int)), this, SLOT(setICRep(int)));

    QHBoxLayout * sliderLayout = new QHBoxLayout;

    QVBoxLayout * fprotr_Layout = new QVBoxLayout;
    fprotr_Layout->addWidget(new QLabel(tr("Force protrusion")));
    fprotr_Layout->addWidget(fprotr_Slider);
    sliderLayout->addLayout(fprotr_Layout);

    QVBoxLayout * evlstiff_Layout = new QVBoxLayout;
    evlstiff_Layout->addWidget(new QLabel(tr("Evl stiffness")));
    evlstiff_Layout->addWidget(evlstiff_Slider);
    sliderLayout->addLayout(evlstiff_Layout);

    QVBoxLayout * evlGratio_Layout = new QVBoxLayout;
    evlGratio_Layout->addWidget(new QLabel(tr("Evl growth ratio")));
    evlGratio_Layout->addWidget(evlGratio_Slider);
    sliderLayout->addLayout(evlGratio_Layout);

    QVBoxLayout * icYolkAttr_Layout = new QVBoxLayout;
    icYolkAttr_Layout->addWidget(new QLabel(tr("Ic-Yolk attraction coeffcient")));
    icYolkAttr_Layout->addWidget(icYolkAttr_Slider);
    sliderLayout->addLayout(icYolkAttr_Layout);

    QVBoxLayout * evlSizeLimit_Layout = new QVBoxLayout;
    evlSizeLimit_Layout->addWidget(new QLabel(tr("Evl size limit")));
    evlSizeLimit_Layout->addWidget(evlSizeLimit_Slider);
    sliderLayout->addLayout(evlSizeLimit_Layout);

    QVBoxLayout * evlGthres_Layout = new QVBoxLayout;
    evlGthres_Layout->addWidget(new QLabel(tr("Evl growth threshold")));
    evlGthres_Layout->addWidget(evlGthres_Slider);
    sliderLayout->addLayout(evlGthres_Layout);

    QVBoxLayout * margRes_Layout = new QVBoxLayout;
    margRes_Layout->addWidget(new QLabel(tr("Margin resistance")));
    margRes_Layout->addWidget(margRes_Slider);
    sliderLayout->addLayout(margRes_Layout);
   
    QVBoxLayout * icAttr_Layout = new QVBoxLayout;
    icAttr_Layout->addWidget(new QLabel(tr("ICIC Attraction")));
    icAttr_Layout->addWidget(icAttr_Slider);
    sliderLayout->addLayout(icAttr_Layout);
    
    QVBoxLayout * icRep_Layout = new QVBoxLayout;
    icRep_Layout->addWidget(new QLabel(tr("ICIC Repulsion")));
    icRep_Layout->addWidget(icRep_Slider);
    sliderLayout->addLayout(icRep_Layout);

    // QVBoxLayout * _Layout = new QVBoxLayout;
    // _Layout->addWidget(new QLabel(tr("")));
    // _Layout->addWidget(_Slider);
    // sliderLayout->addLayout(_Layout);
   
    setLayout(sliderLayout);
    
    setMinimumSize(300,300);

    setWindowTitle(tr("Epiboly Param"));

  }

  CustomParamWidget::~CustomParamWidget() { }

  //
  // Hints:
  //

  QSize CustomParamWidget::minimumSizeHint() const { return QSize(50, 50); }

  QSize CustomParamWidget::sizeHint() const { return QSize(100, 300); }

  void CustomParamWidget::initSliders()
  {
    fprotr_Slider            -> setValue( 10000.0 * param->protrusionNode[0].force );
    evlstiff_Slider          -> setValue( param->customParam[0].evlStiffness );
    evlGratio_Slider         -> setValue( 1000.0 * (param->customParam[0].evlLateralGrowthRatio - 1.0) );
    icYolkAttr_Slider        -> setValue( .01 * param->customParam[0].cellsYolkAttractionCoefficient );
    evlSizeLimit_Slider      -> setValue( 2000.0 * param->customParam[0].evlRadiusLimit );
    evlGthres_Slider         -> setValue( 100.0 * param->customParam[0].evlGrowthThreshold );
    margRes_Slider           -> setValue( 1000.0 * param->customParam[0].marginResistance );
    icAttr_Slider            -> setValue( .01 * param->adhesionNodes[0].k_adh );
    icRep_Slider             -> setValue( .01 * param->mechaParams[0].repulsionCoefficient[4] );  // 3*1+1 (celltype = 1, mesenchymal)
  }

  void CustomParamWidget::setFProtrusion(int val)
  {
    printf("force protrusion %lf\n", (double)val/10000.0);
    param->protrusionNode[0].force = (double) val/10000.0;
  }

  void CustomParamWidget::setEvlStiffness(int val)
  {
    printf("evl stiffness %lf\n", (double)val);
    param->customParam[0].evlStiffness = (double) val; 
  }

  void CustomParamWidget::setEvlGrowthRatio(int val)
  {
    printf("evl growth ratio %lf\n", 1.0+(double)val/1000.0);
    param->customParam[0].evlLateralGrowthRatio = 1.0+(double) val/1000.0;
  }

  void CustomParamWidget::setICYolkAttr(int val)
  {
    printf("ic yolk attraction coefficient %lf\n", (double)val/.01);
    param->customParam[0].cellsYolkAttractionCoefficient = (double) val/.01;
  }

  void CustomParamWidget::setEvlSizeLimit(int val)
  {
    printf("Evl limit radius %lf\n", (double)val/2000.0);
    param->customParam[0].evlRadiusLimit = (double) val/2000.0;
  }

  void CustomParamWidget::setEvlGrowthThreshold(int val)
  {
    printf("Evl growth threshold %lf\n", (double)val/100.0);
    param->customParam[0].evlGrowthThreshold = (double) val/100.0;
  }

  void CustomParamWidget::setMarginResistance(int val)
  {
    printf("marginResistance %lf\n", (double)val/1000.0);
    param->customParam[0].marginResistance = (double) val/1000.0;
  }

  void CustomParamWidget::setICAttr(int val)
  {
    printf("IC attraction %lf\n", (double)val/.01);
    param->adhesionNodes[0].k_adh = (double) val/.01;
  }

  void CustomParamWidget::setICRep(int val)
  {
    printf("IC repulsion %lf\n", (double)val/.01);
    param->mechaParams[0].repulsionCoefficient[4] = (double) val/.01;
  }

}
