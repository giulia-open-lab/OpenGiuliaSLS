# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:10:08 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import sys
import numpy as np


class EventDrivenSimulation:
    def __init__(self, init, max_time_us):
        
        # Size of the timex piles
        self.max_time_us = max_time_us
        self.decision_making_interval_us  = None  # This updated when calling the run method
        self.minimum_messaging_resolution = 1e2
        self.messaging_time_resolution = self.max_time_us/self.minimum_messaging_resolution   
        
        # init = 0 when we want to disable the event driven simulation 
        if init == 0:
            self.current_time_us = -1
            self.tti = -1
            self.event_id = -1
            self.events_in_the_pile = -1
            pile_depth = 1000

        # init = 1 when we want to enable the event driven simulation    
        elif init == 1:
            self.current_time_us = 0
            self.tti = 0
            self.max_timex_time = self.max_time_us
            self.event_id = 0
            self.events_in_the_pile = 0
            pile_depth = 10000
            
        # Events pile
        # This pile contains the events that will be process via feval
        self.pile_events = [None] * pile_depth
        
        # Events pile allocation
        # This pile indicates whether an entry of the events_pile is used or not
        # Column 0 indicates the position in the event_pile list
        # Column 1 indicates whther it is used or not
        self.pile_events_allocation = np.zeros((pile_depth, 1), dtype=int)
        
        # Events order pile
        # This pile contains the event time and its position in the events_pile
        # Column 0 indicates the event time
        # Column 1 indicates the position in the event_pile list
        # Column 2 indicates the number of events with the same event time
        self.pile_events_order = np.full((pile_depth, 3), -1, dtype=int)
        self.pile_events_order[:, 2] = 0  # Set the third column to zero
        

    def plot_hello(self, a=0, b=0):
        np.disp("Hello! = " + str(a) + "bye = " + str(b))
        return 5


    def find_first_false(self, bool_array):
        
        if not np.any(~bool_array):
            # All values are True, return an indication that no False exists
            return -1
        
        # np.argmax will return the index of the first True in inf_mask
        return np.argmax(~bool_array)
    
    
    def find_first_nan(self, array):
        
        # Create a boolean mask where True corresponds to elements equal to -1
        mask = (array == -1)
        
        # Check if there's at least one -1 in the array
        if mask.any():
            # Use np.argmax to find the first True in the mask, which corresponds to the first -1 in the array
            return np.argmax(mask)
        else:
            # Return -1 or another appropriate value if no -1 is found
            return -1


    def add_event(self, timing_from_now_us, event_object, event_description, *args):
        
        # Finding free locations in pile_events and pile_events_order
        free_location_in_pile_events = self.find_first_false(self.pile_events_allocation)
        free_locations_in_pile_events_order = self.find_first_nan(self.pile_events_order[:,1])
        
        # Add event if there are free positions
        if free_location_in_pile_events != -1 and free_locations_in_pile_events_order != -1:
            
            # Increase event ID
            self.event_id += 1
            self.events_in_the_pile += 1
            
            # Assign the free location from pile_events
            self.pile_events[free_location_in_pile_events] = [self.event_id, event_object, event_description, args]
            self.pile_events_allocation[free_location_in_pile_events] = 1  # Mark as used

            # Assign the first free location from pile_events_order
            new_event_time = round((self.current_time_us + timing_from_now_us) * 1e4) / 1e4 # Set time of event
            self.pile_events_order[free_locations_in_pile_events_order, 0] = new_event_time
            self.pile_events_order[free_locations_in_pile_events_order, 1] = free_location_in_pile_events
        
            # Maintain ordering consistency by time
            events_with_the_same_event_time = np.abs(self.pile_events_order[:, 0] - new_event_time)  < 1e-6 # Find events with the same event time
            self.pile_events_order[events_with_the_same_event_time, 0] = new_event_time # Make sure that all have the same time
            self.pile_events_order[free_locations_in_pile_events_order, 2] = max(self.pile_events_order[events_with_the_same_event_time, 2]) + 1 # Update number of events with same event time
            
            return self.pile_events_order[free_locations_in_pile_events_order,1]
        
        else:
            np.disp('Error: All positions of pile_events are occupied')
            sys.exit(0)  


    def remove_event(self, index):
        
        if self.events_in_the_pile > 0:

            # Clear the event details from pile_events
            self.pile_events[index] = [np.inf, None, None]
        
            # Update the pile allocation, marking this position as empty
            self.pile_events_allocation[index] = 0
        
            # Search for the event in the pile_events_order and update
            index2 = np.where(self.pile_events_order[:, 1] == index)[0]
            if index2.size > 0:
                self.pile_events_order[index2, :] = [-1, -1, 0]
            else :
                print('Error: The event that has to be removed it is not in the pile_events')
                sys.exit(0) 
        
            # Decrease the count of events in the pile
            self.events_in_the_pile -= 1
        
            return 1
            
        else:
            print('Error: No events to remove')
            sys.exit(0)

            return 0
        
        
    def run(self, step_index, decision_making_interval_us):
        
        self.step_index = step_index
        self.decision_making_interval_us = decision_making_interval_us 
        
        while self.events_in_the_pile > 0 \
            and self.current_time_us < self.max_time_us \
            and self.current_time_us - self.step_index * self.decision_making_interval_us  < self.decision_making_interval_us :
            
            # Create a mask that is True for all values except -1
            valid_event_times = self.pile_events_order[self.pile_events_order[:,0] >= 0, 0]            
            
            # Get the current minimum time event
            current_min_time = np.min(valid_event_times)
            events_with_current_min_time = np.where(np.abs(self.pile_events_order[:, 0] - current_min_time) < 1e-6)[0]
            
            # Select the event with the smallest ID at the current minimum time
            min_ID = np.min(self.pile_events_order[events_with_current_min_time, 2])
            event_with_current_min_time = events_with_current_min_time[self.pile_events_order[events_with_current_min_time, 2] == min_ID][0]
            
            # Set current time
            self.current_time_us = self.pile_events_order[event_with_current_min_time, 0]
            if self.current_time_us >= self.max_time_us :
                break
            if self.current_time_us - self.step_index * self.decision_making_interval_us >= self.decision_making_interval_us :
                break         
            
            # Process the next event
            self.location_in_pile_events = self.pile_events_order[event_with_current_min_time, 1]
            
            # Process the event using a callback function and handle the event's return value
            object_name = self.pile_events[self.location_in_pile_events][1]
            method_name = self.pile_events[self.location_in_pile_events][2]
            arguments = self.pile_events[self.location_in_pile_events][3]
            
            # Dynamically get the method and call it with the list of arguments
            if object_name is None:
                timex_value = getattr(method_name)(*arguments)
            else:
                timex_value = getattr(object_name, method_name)(*arguments)
            
            if timex_value == -1:
                self.remove_event(self.location_in_pile_events)
            else:
                new_event_time = self.current_time_us + timex_value
                self.pile_events_order[event_with_current_min_time, 0] = new_event_time
                
                # Ensure consistent ordering for events rescheduled to the same time
                events_with_the_same_event_time = np.abs(self.pile_events_order[:, 0] - new_event_time)  < 1e-6 # Find events with the same event time
                self.pile_events_order[events_with_the_same_event_time, 0] = new_event_time # Make sure that all have the same time
                self.pile_events_order[event_with_current_min_time, 2] = max(self.pile_events_order[events_with_the_same_event_time, 2]) + 1 # Update number of events with same event time                
