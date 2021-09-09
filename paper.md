---
title: 'DRSP-Sim: A Simulator for Ride-Sharing with Pooling: Joint Matching, Pricing, Route Planning, and Dispatching'
tags:
  - Ridesharing
  - Deep Reinforcement Learning
  - Shared Mobility
authors:
  - name: Marina Haliem
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Vaneet Aggarwal
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Bharat Bhargava
    orcid: 0000-0003-0872-7098
    affiliation: 1
affiliations:
 - name: Purdue University, West Lafayette, IN
   index: 1
date: 10 September 2021
bibliography: paper.bib
---

# Summary

Ridesharing is an emerging mode of transportation currently having a deep impact on the personal transportation industry. Although several ride-sharing algorithms have been developed, real-time evaluations remain the greatest challenge of such approaches. Also, algorithms of customer-vehicle matching, route planning, pricing and dispatching have been developed; however, these sub-problems tend to be studied separately and a complete integrated simulator that considers pooling is still lacking. In this paper, (1) we develop a real-time Dynamic RideSharing simulator with Pooling (DRSP-Sim) for evaluating ridesharing algorithms integrated into one simulator, and (2) we provide benchmarks for vehicle-customer matching, route planning, pricing and dispatching to test a wide range of scenarios encountered in the real world. Our work enables real-time evaluations and provides guidance for designing and evaluating future ridesharing algorithms.

# Statement of need

Ride-Sharing (RS) has the potential of transforming urban mobility by providing personal convenience to each customer by accommodating their unique preferences. However, real-time evaluation of various ridesharing algorithms integrated together has been a real challenge due to the lack of a complete simulator that can support all the sub-problems at the same time. Vehicle-customer matching [@tafreshian2020trip; @alonso2017demand], vehicle route-planning [@MOLENBRUCH201758; @schonberger2017scheduling], ride pricing \citep{zhang2020pricing} and vehicle dispatching [@deep_pool; @fleet_oda] are examples of the RS sub-problems that has been widely studied in literature; however, they have been studied separately or in combination of two. We present the first RS simulator that integrates all of these together, provides a baseline for each, and allows developers to plug-and-play algorithms to tackle any of the sub-problems while defaulting the rest to the provided benchmark. This, in turn, will greatly facilitate cross-study evaluations of any existing ridesharing algorithms, while providing guidance for future research.

#Simulator Architecture

![Overall architecture of the simulator.\label{arch}](Architecture00.pdf){ width=45% }
Existing simulators [@abs-1710-05465; @article19] either don't support pooling, lack the integration of the RS sub-problems, or generate artificial data and lack the support to real-world data and city maps. In contrast, DRSP-Sim combines the RS sub-problems and uses real-world data and maps. We construct a region graph of the New York Metropolitan area obtained from Open-StreetMap [@osm], along with a public dataset of taxi trips in NY (15 million trips) [@10].
Figure \autoref{arch} shows the main components of our simulator. We assume that the central control unit is responsible for: (1) maintaining the states such as current locations, current capacity, destinations, etc., for all vehicles. These states are updated in every time step based on the dispatching and matching decisions. (2) The control unit also has some internal components that help manage the ride-sharing environment such as: (a) the estimated time of arrival (ETA) model used to calculate and update the estimated arrival time. (b) The Open Source Routing Machine (OSRM) model used to generate the vehicle's optimal trajectory to reach a destination, and (c) the (Demand Prediction) model used to calculate the future anticipated demand. We adopt these three models from [@fleet_oda]. We note that our simulator will support any algorithm for any sub-problem.
![Code architecture of the simulator. \label{modular}](codeArch1.pdf){ width=45% }

Here, we explain the work flow of DRSP-Sim using our benchmark algorithms provided for each sub-problem. For every time step, first, the ride requests are input to the system along with the heat map for supply and demand (which involves demand prediction in the near future). Then, based on the predicted demand, vehicles adopt a dispatching policy using DQN [@9507388], where they get dispatched to zones with anticipated high demand. This step not only takes place when vehicles first enter the market, but also when they experience large idle durations. Then, each vehicle receives the updated environment state from the control unit and performs a vehicle-passenger(s) matching approach, where one request (or more) gets assigned to each vehicle based on its maximum passenger capacity. Next, communicating with the Price Estimation model, each vehicle calculates the corresponding initial pricing associated with each request. Afterward, each vehicle executes its matching optimizer module to perform an insertion-based route planning.
Using the demand-aware pricing provided by our simulator [@haliem2020distributed], vehicles weigh their utility based on the potential hotspot locations, and propose new pricing for the customer. 
Figure \autoref{modular} shows the file organization of our code base which makes it easy to navigate through the simulator. We provide a \textit{configuration} module which allows the user to specify which features to enable (e.g., pooling, pricing, dispatching, matching) as well as which algorithms to use for each. Besides, the simulator allows a \textit{Logging} service that logs every event that takes place during the simulation runtime (e.g, customer pickup/dropoff).


# References
