 /*
 * Copyright (C) 2015 by Julien Delile and Matthieu Herrmann
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

#ifndef _INTERFACES_H_2013_08
#define _INTERFACES_H_2013_08

// Include standard:
#include <map>
#include <set>
#include <iostream>

namespace isf{

  // Forward declarations:
  class Broker;
  class Consumer;
  class ProducerFactory;

  /** ProducerParameter class.
  * The ProducerParameter class contains the pair of parameters that are specified in the client's consumer class to identify the
  * required producer classes and their period of action.
  */
  class ProducerParameter{
    public:
      ProducerFactory*  prFact;
      int               period;
  };

  /** NotifyParameter class.
  * The NotifyParameter class contains the pair of parameters that are passed in an event send by a broker to a 
  * consumer.
  * @param[in] brokerID       The broker ID, stored as pointer of ProducerFactory
  * @param[in] step           The time step associated with the notification
  * @param[in] errorCode      The errorCode of the notification, used optionally
  */
  class NotifyParameter{
    public:

      NotifyParameter(){
        errorCode = 0;
      };

      ProducerFactory*  brokerID;
      int               step;
      int               errorCode;
  };
  
  /** ProducerConfig class.
  * The ProducerConfig class contains the set of period that will trigger the execution of the producer. If this
  * set contains the value 10, the producer will execute every 10 time steps.
  */
  class ProducerConfig{
    public:
      /** Default ProducerConfig constructor. */
      ProducerConfig(){};

      /** Overloaded ProducerConfig constructor. 
      */
      ProducerConfig( std::set<int> config_):
        config(config_){};

      // Configuration data:
      /** Is used to store the set of period at which the producer is executed.*/
      std::set<int> config;
  };

  /** BrokerConfig class.
  * The BrokerConfig class stores maps between the consumer pointers and the period at which they need to be
  * notified.
  */
  class BrokerConfig{
    public:
      /** BrokerConfig constructor.*/
      BrokerConfig( std::map< Consumer*, std::set<int> > config_):
        config(config_){};

      // Configuration data:
      /** Is used to store the map between the consumer pointers and the period at which the need to be notified.*/
      std::map< Consumer*, std::set<int> > config;
  };

  /** ConsumerConfig class.
  * The ConsumerConfig class store the map between the broker identifiers and the broker pointers that is required 
  * by the consumer to access the buffered data.
  */
  class ConsumerConfig{
    public:
      /** Default ConsumerConfig constructor.*/
      ConsumerConfig(){};

      /** Overloaded ConsumerConfig destructor.*/
      ConsumerConfig( std::map< ProducerFactory*, Broker* > config_):
        config(config_){};

      // Configuration data:
      /** Is used to store the map between the broker identifiers and the broker pointers that is required 
      * by the consumer to access the buffered data.
      */
      std::map< ProducerFactory*, Broker* > config;
  };

}
#endif
