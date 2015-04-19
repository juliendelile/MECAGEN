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

#include "window.hpp"
#include "controller.hpp"

namespace mg{

  Window::Window(
      isf::Controller<MetaParam<HOST>, Param_Host, State_Host > ** _c,
      MetaParam<HOST> * _mp, Param_Host * _p, State_Host * _s, QApplication *parent):
    c(_c),
    metaParam(_mp),
    param(_p),
    state(_s)
  {

    // Specify an OpenGL 3.2 format using the Core profile.
    // That is, no old-school fixed pipeline functionality
    QGLFormat glFormat;
    // glFormat.setVersion( 3, 2 );
    // glFormat.setProfile( QGLFormat::CoreProfile );
    glFormat.setSampleBuffers( true );

    // Create the openGL widget:
    glWidget     = new GLWidget(metaParam, param, glFormat);

    disConWidget = new DisplayControlWidget(glWidget, this, param);
    disConWidget->hide();


    //pass param to custom widget so that it can modify it with the adapted setXé method
    // customParamWidget = new CustomParamWidget(this, param);
    // customParamWidget->hide();

    recordingWidget = new RecordingWidget(glWidget, this);
    recordingWidget->hide();

    playButton = new QPushButton("Start", this);
    pauseButton = new QPushButton("Pause", this);
    pauseButton->setEnabled(false);
    plusOneButton = new QPushButton("Next Timestep", this);
    disConButton = new QCheckBox("Display Control", this);
    disConButton->setAutoExclusive(false);
    // disConButton->setChecked(true);
    // customParamButton = new QRadioButton("Simulation Param", this);
    // customParamButton->setAutoExclusive(false);
    recordSnapButton = new QCheckBox("Snapshot Tools", this);
    recordSnapButton->setAutoExclusive(false);

    recordStateParamButton = new QPushButton("Save Current State (xml files)", this);
    // recordStateParamButton->setAutoExclusive(false);

    connect(playButton, SIGNAL(clicked(bool)), this, SLOT(callStart()));
    connect(pauseButton, SIGNAL(clicked(bool)), this, SLOT(callPause()));
    connect(plusOneButton, SIGNAL(clicked(bool)), this, SLOT(callPlusOne()));
    connect(disConButton, SIGNAL(toggled(bool)), this, SLOT(showHideDisplayControl()));
    //connect(customParamButton, SIGNAL(toggled(bool)), this, SLOT(showHideCustomParam()));
    connect(recordSnapButton, SIGNAL(toggled(bool)), this, SLOT(showHideRecording()));
    connect(recordStateParamButton, SIGNAL(clicked(bool)), this, SLOT(saveStateParam()));

    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(playButton);
    buttonLayout->addWidget(pauseButton);
    buttonLayout->addWidget(plusOneButton);

    QHBoxLayout *radiobuttonLayout = new QHBoxLayout;
    radiobuttonLayout->addWidget(disConButton);
    // radiobuttonLayout->addWidget(customParamButton);
    radiobuttonLayout->addWidget(recordSnapButton);
    radiobuttonLayout->addWidget(recordStateParamButton);
    
    QVBoxLayout * leftLayout = new QVBoxLayout;
    leftLayout->addLayout(buttonLayout);
    leftLayout->addWidget(glWidget);
    leftLayout->addLayout(radiobuttonLayout);
  
    // QHBoxLayout *mainLayout = new QHBoxLayout;
    // mainLayout->addLayout(leftLayout);

    setLayout(leftLayout);

    setMinimumSize(300,300);

    setWindowTitle(tr("MecaGen Viewer"));

    // Open display control panel by default
    disConButton->toggle();
  }

  // Destructor
  Window::~Window() throw () {
    
  

    
  }

  void Window::closeEvent(QCloseEvent *event)
  {
    // event->ignore();
    delete playButton;
    delete pauseButton;
    delete plusOneButton;
    delete disConButton;
    // delete customParamButton;
    delete recordSnapButton;
    delete disConWidget;
    // delete customParamWidget;
    delete recordingWidget;
    delete glWidget;
    
    ((QApplication*)(this->parent()))->exit();
  }

  void Window::initGUI(int ts, int te){
    // customParamWidget->initSliders();
    
    time_start  = ts;
    time_end    = te;
  }

  void Window::keyPressEvent(QKeyEvent *e)
  {
    if (e->key() == Qt::Key_Escape) { close(); }
    else if (e->key() == Qt::Key_R) { glWidget->scale += glWidget->scale*0.1; }
    else if (e->key() == Qt::Key_F) { glWidget->scale -= glWidget->scale*0.1; }
    else if (e->key() == Qt::Key_G) { 
      if(!isFullScreen()){
        showFullScreen();
        playButton->hide();      
        pauseButton->hide();      
        plusOneButton->hide();      
        disConButton->hide();      
        customParamButton->hide();      
        recordSnapButton->hide();      
        recordStateParamButton->hide();      
      }
      else{
        showMaximized();
        playButton->show();      
        pauseButton->show();      
        plusOneButton->show();      
        disConButton->show();      
        customParamButton->show();      
        recordSnapButton->show();      
        recordStateParamButton->show();
      }
    }    
    else{ 
      QWidget::keyPressEvent(e); 
    }
  }

  void Window::wheelEvent (QWheelEvent * event)
  {
    event->delta() > 0 ? glWidget->scale += glWidget->scale*0.1 : glWidget->scale -= glWidget->scale*0.1;
  }

  //
  // Gui buttons slots:
  //

  void Window::callStart(){

    //return if the model is running
    if(!((*c)->isPause() || (*c)->isOver())){
      return;
    }

    playButton->setText("Restart");
    playButton->setEnabled(false);

    pauseButton->setText("Pause");
    pauseButton->setEnabled(true);

    plusOneButton->setEnabled(false);
    

    //change seeds
    // int seedGauss = (int)(400000.0 * rand()/ (double) RAND_MAX);
    // int seedUnif = (int)(400000.0 * rand()/ (double) RAND_MAX);
    
    // param->load_RandomGaussian(time(0));
    // param->load_RandomUniform(time(0)+67);

    param->load_RandomGaussian(89);
    param->load_RandomUniform(67);

    
    (*c)->setParam(param);
    
    // state->init(param);
    (*c)->setState(state);
    (*c)->setTimeInterval(time_start, time_end);
    (*c)->start();
  }

  void Window::callPause(){

    if(pauseButton->text() == "Play"){ // Set to play
      pauseButton->setText("Pause");
      playButton->setEnabled(false);
      plusOneButton->setEnabled(false);
    }
    else{
      pauseButton->setText("Play");   // set to pause
      playButton->setEnabled(true);
      plusOneButton->setEnabled(true);
    }

    (*c)->playPause();
  }

  void Window::callPlusOne(){
    // std::cout << "plus one "<< std::endl;

    playButton->setText("Restart"); // in case +1 is hit first

    pauseButton->setEnabled(false);

    (*c)->setParam(param);
    (*c)->setTimeInterval((*c)->getCurrentTimeStep(), (*c)->getCurrentTimeStep());
    (*c)->start();
  }

  void Window::showHideDisplayControl(){
    if(disConButton->isChecked()){
      disConWidget->show();
    }
    else{
      disConWidget->hide();
    }
  }

  void Window::showHideCustomParam(){
    if(customParamButton->isChecked()){
      customParamWidget->show();
    }
    else{
      customParamWidget->hide();
    }
  }

  void Window::showHideRecording(){
    if(recordSnapButton->isChecked()){
      recordingWidget->show();
    }
    else{
      recordingWidget->hide();
    }
  }

  void Window::saveStateParam(){

    QString address = QFileDialog::getExistingDirectory(
                              this, 
                              tr("Select/create directory"), 
                              "/tmp",
                              QFileDialog::ShowDirsOnly
                              );
    if(address.isNull()){
      printf("Cancelled\n");
      return;
    }

    QString state_address = address+"/state_archive.xml";
    QString param_address = address+"/param_archive.xml";
    QString metaparam_address = address+"/metaparam_archive.xml";

    State_Host      state = (*c)->getState();

    save< State_Host >(state, state_address.toStdString().c_str());
    save< Param_Host >((*param), param_address.toStdString().c_str());
    save< MetaParam<HOST> >((*metaParam), metaparam_address.toStdString().c_str());
  }  

  //
  // Gui sliders slots:
  //

  // void Window::setAdhHomotypic(int val){
  //   param->w_adh_homotypic[0] = (double) val;
  //   param->displayOnConsole();
  // }

  // void Window::setAdhHeterotypic(int val){
  //   param->w_adh_heterotypic[0] = (double) val;
  //   param->displayOnConsole();
  // }

  // void Window::setRepulsion(int val){
  //   param->w_rep[0] = (double) val;
  //   param->displayOnConsole();
  // }

  // void Window::setRigidity(int val){
  //   param->k_rig[0] = (double) val;
  //   param->displayOnConsole();
  // }

  // void Window::setGabCoeff(int val){
  //   param->gab_coeff[0] = (double) val / 100.0;
  //   param->displayOnConsole();
  // }

  // void Window::setCmax(int val){
  //   param->c_max[0] = (double) val / 100.0;
  //   param->displayOnConsole();
  // }

  void Window::setDamping(int val){
    param->globalDamping[0] = (double) val;
    // param->displayOnConsole();
  }

}
