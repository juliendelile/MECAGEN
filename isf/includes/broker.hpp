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

#ifndef _BROKER_H_2013_08
#define _BROKER_H_2013_08

// Project definition
#include "interfaces.hpp"

// Standard C definition (size_t)
#include <cstddef>


namespace isf {

  // Forward declarations:
  template<typename T> class EventManager;

  /** The Broker Class.
   * The Broker is an intermediary module which is used to transfer data from a single Producer module
   * to any enquiring Consumer modules (it allows transfers between N producers and M consumers).
   * It contains a customizable double swapping buffer and an EventManager which receives notification
   * from the consumers.*/
  class Broker {

    public:
      /** Broker Constructor.
       *  The constructor is public as it used by Controller_Impl in SetConsumers, but it shall
       *  remain hidden from the client code.
       *  @param[in] conf               The std::map containing the consumers pointers as key and their
       *  set of period of notification as values (std::set<int>). It should be noted that the
       *  Controller_Impl passes the whole list of Consumers specified by the client, and not the
       *  specific list associated by the constructed broker (this should be changed...)
       *  @param[in] allocationSize     The size of the buffer required by the associated producer.
       *  @param[in] brid               The broker ID.
       */
      Broker(BrokerConfig conf, size_t allocationSize, ProducerFactory* brid);

      /** Broker Destructor.*/
      ~Broker() throw ();

      // Event API:
      /** Add a Event in the broker's eventmanager which calls the void doReleaseBuffer() internal method.
       * This method is called by a consumer to notify the broker that it has finished to work in the comsumerBuffer.
       */
      void releaseBuffer(); // for consumer

      //public API:
      /** Exchange the both swapping buffers' pointers. This method is called by the associated producer
       * when it has finished its data processing. To allow the swapping, the broker first checks that all
       * the associated consumers have finished to use the previous traded data (via the notificationCounter
       * variable). The broker may blocks the simulation thread here, if some consumers have not finished to
       * read/process their data. Once the pointers have swapped, the broker notifies its associated
       * consumers that newer are available for processing.
       * @param[in] step     The time step at which the data is provided
       */
      void swapBuffers(int step);

      /** Return the pointer pointing to the buffer instance accessible by the producer for writing.*/
      void * getProducerBuffer(); //for producer in doProcessing

      /** Return the pointer pointing to the buffer instance accessible by the consumer for reading.*/
      void * getConsumerBuffer(); //for consumer in receiveNotification

    private:
      // Internal Data:
      /** Is used to store the map between Consumers' pointers and their period of action. */
      BrokerConfig              brokerConfig;

      /** Is used to store the current number of consumers that have not finished to process their
       * data. This value is set to be equal to the number of associated consumers before the broker
       * notifies them, and, when it equals 0 again, the broker can swap its buffer.
       */
      int                       notificationCounter;

      /** Is used to store the Events send by the consumer thread via the broker API. These events are
       * read by the broker thread (i.e. simulation thread). */
      EventManager<Broker>*     em;

      /** Is used to store the ID of the broker. The pointer of its associated producer's ProducerFactory
       * is used. */
      ProducerFactory*          brokerID;

      /* Is used to store the current adress of the producer buffer.*/
      void *                    producerBuffer;

      /* Is used to store the current adress of the consumer buffer.*/
      void *                    consumerBuffer;

      // Internal API:
      /** Decrement the notificationCounter by one. */
      void doReleaseBuffer();     //void doNotifyBroker();

      /** Allocate the memory of the two broker's buffers.
       * @param[in] size       The size of the buffer's memory in bytes.
       */
      void doAllocation(size_t size);

      /** Call the consumers' public method "notify" when the swap have been executed.
       * @param[in] step       The time step at which the data is provided.
       */
      void notifyConsumers(int step);
  };

} // End namespace
#endif
