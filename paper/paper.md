---
title: 'DRSP-Sim: A Simulator for Ride-Sharing with Pooling: Joint Matching, Pricing, Route Planning, and Dispatching'
tags:
  - Ridesharing
  - Deep Reinforcement Learning
  - Shared Mobility
authors:
  - name: Marina Haliem
    orcid: 0000-0002-9782-6591
    affiliation: 1
  - name: Vaneet Aggarwal
    orcid: 0000-0001-9131-4723
    affiliation: 1
  - name: Bharat Bhargava
    orcid: 0000-0003-3803-8672
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

Ride-Sharing (RS) has the potential of transforming urban mobility by providing personal convenience to each customer by accommodating their unique preferences. However, real-time evaluation of various ridesharing algorithms integrated together has been a real challenge due to the lack of a complete simulator that can support all the sub-problems at the same time. Vehicle-customer matching [@tafreshian2020trip; @alonso2017demand], vehicle route-planning [@MOLENBRUCH201758; @schonberger2017scheduling], ride pricing [@zhang2020pricing] and vehicle dispatching [@deep_pool; @fleet_oda] are examples of the RS sub-problems that has been widely studied in literature; however, they have been studied separately or in combination of two. We present the first RS simulator that integrates all of these together, provides a baseline for each, and allows developers to plug-and-play algorithms to tackle any of the sub-problems while defaulting the rest to the provided benchmark. This, in turn, will greatly facilitate cross-study evaluations of any existing ridesharing algorithms, while providing guidance for future research.

# Simulator Architecture

![Overall architecture of the simulator.\label{arch}](Architecture00.pdf){ width=60% }

Existing simulators [@abs-1710-05465; @article19] either don't support pooling, lack the integration of the RS sub-problems, or generate artificial data and lack the support to real-world data and city maps. In contrast, DRSP-Sim combines the RS sub-problems and uses real-world data and maps. We construct a region graph of the New York Metropolitan area obtained from Open-StreetMap [@OSM], along with a public dataset of taxi trips in NY (15 million trips) [@10].
\autoref{arch} shows the main components of our simulator. We assume that the central control unit is responsible for: (1) maintaining the states such as current locations, current capacity, destinations, etc., for all vehicles. These states are updated in every time step based on the dispatching and matching decisions. (2) The control unit also has some internal components that help manage the ride-sharing environment such as: (a) the estimated time of arrival (ETA) model used to calculate and update the estimated arrival time. (b) The Open Source Routing Machine (OSRM) model used to generate the vehicle's optimal trajectory to reach a destination, and (c) the (Demand Prediction) model used to calculate the future anticipated demand. We adopt these three models from [@fleet_oda]. We note that our simulator will support any algorithm for any sub-problem.

![Code architecture of the simulator. \label{modular}](codeArch.png){ width=60% }

Here, we explain the work flow of DRSP-Sim using our benchmark algorithms provided for each sub-problem. For every time step, first, the ride requests are input to the system along with the heat map for supply and demand (which involves demand prediction in the near future). Then, based on the predicted demand, vehicles adopt a dispatching policy using DQN [@9507388], where they get dispatched to zones with anticipated high demand. This step not only takes place when vehicles first enter the market, but also when they experience large idle durations. Then, each vehicle receives the updated environment state from the control unit and performs a vehicle-passenger(s) matching approach, where one request (or more) gets assigned to each vehicle based on its maximum passenger capacity. Next, communicating with the Price Estimation model, each vehicle calculates the corresponding initial pricing associated with each request. Afterward, each vehicle executes its matching optimizer module to perform an insertion-based route planning.
Using the demand-aware pricing provided by our simulator [@haliem2020distributed], vehicles weigh their utility based on the potential hotspot locations, and propose new pricing for the customer. 
\autoref{modular} shows the file organization of our code base which makes it easy to navigate through the simulator. We provide a \textit{configuration} module which allows the user to specify which features to enable (e.g., pooling, pricing, dispatching, matching) as well as which algorithms to use for each. Besides, the simulator allows a \textit{Logging} service that logs every event that takes place during the simulation runtime (e.g, customer pickup/dropoff).

## Code Availability and Documentation

The code for this comprehensive simulator can be seen at
\url{https://github.itap.purdue.edu/Clan-labs/Dynamic-RideSharing-Pooling-Simulator}, where the details on the data and the simulator setup are provided. We provide the pre-processed data of the NYC taxi trips at \url{https://purr.purdue.edu/projects/ridesharing/files}. In addition, instructions on how to generate this data from scratch is also available. Besides the integration of RS sub-problems and the ability to plug-in any such algorithm, DRSP-Sim provides a wide-range of flexibilites to enable conducting various RS scenarios. Some of these are: the ability to enable/disable the pooling feature, decide how many vehicles to populate in the simulation and how many of them adopt the DQN dispatch policy, store trained networks and replay memory, and control the map-related variables and DQN training hyper-parameters.

# Benchmarks
DRSP-Sim supports pooling, which allows vehicles to pickup more than one customer at the same time. This adds more complexities to the ridesharing scenario where the route planning needs to be optimized to accommodate all customers. Matching, pricing, and dispatching algorithms need to be devised such that they take pooling into consideration.

## Matching and Route Planning:
\textbf{Non-dynamic Greedy Matching: } In this algorithm, customer rides get assigned greedily to the nearest vehicle in its vicinity as long as the vehicle's capacity can accommodate. In this case, matching only happens at the beginning of every time-step for idle vehicles only. No dynamic matching takes place, meaning vehicles that are already assigned a route to accommodate one or more rides, aren't assigned any new rides until they become idle again.

\textbf{Dynamic Insertion-based matching and route planning: } In this algorithm, partially occupied vehicles are also considered while allocating new rides at the beginning of every time-step. In that case, when the partially occupied vehicle gets assigned new rides, it performs an insertion-based route planning mechanism to decide on the best route to take that would accommodate both the new customers and the on-board customers. During this route-planning, each vehicle identifies the potential rides in its vicinity, and then decides on which rides to accept according to their corresponding insertion cost  to its current route [@9507388].

## Dispatching:
\textbf{Destination-driven dispatching: } In this dispatching approach, vehicles only get dispatched according to the next stop in their route which is composed of a sequence of pickup and drop-off locations to serve all of its on-board customers. In that case, vehicles never get dispatched when they are idle and can't find new rides to serve.

\textbf{Distributed DQN Dispatching: } In this approach, in addition to the destination-driven dispatching of vehicles, a distributed DQN dispatch policy is utilized to re-balance idle vehicles to areas of predicted high demand and profits over the city, where they can better serve the demand and maximize their profits [@9507388].

## Pricing:
\textbf{Pooling Pricing: } In our simulator, we provide two approaches to price the rides. One is to just accept the fare of the ride that is calculated according to these factors: (1) The total trip distance to serve this particular ride, (2) Number of customers who share travelling a trip distance, (3) The cost for fuel consumption associated with this trip, and (4) The waiting time experienced by the customer [@haliem2020distributed].

\textbf{Demand-Aware Pricing: } In this approach, each vehicle  gains the necessary insight about how the supply-demand is distributed over the city through Q-network, and thus can make informed decisions on the pricing strategy that can yield him a higher profit. The implemented approach follows  [@haliem2020distributed, @9507388], where the vehicles propose a slightly higher price in case of having to drive to an area of predicted low demand.

# Conclusion
In this work, we provide DRSP-Sim: a comprehensive simulator for the ride-sharing service with flexibilities of matching, dispatching, pricing, demand prediction, and routing, where the research on any component can be tested with the overall system. The simulator uses the NY city map, and the NYC taxi-cab data set retrieved from [@10]. DRSP-Sim can run with different datasets, as well as help evaluating and testing of improved algorithms. The simulator has been used as a base for multiple analysis for ride-sharing evaluations [@manchella2020flexpool; @singh2019distributed; @HaliemAB20; @PassGoodPool], which exploit multi-hop passenger transportation, adaptivity in algorithms based on diurnal demand patterns, and combined goods and passenger transportation.

# References
