# Simulation data and properties

<!-- I think it's worth calling out LaserFrame and PropertySet as features, since this is where the nuts and bolts seem to happen (page could use a better title)

LaserFrame: basically a database that manages the dynamically allocated data for agents and nodes

PropertySet: stores the agent properties in a dictionary-like object (such as infection status, region, exposure timers, etc) -->

LASER is designed to handle large populations with high numbers of independent agents. In order to track agents and their properties, LASER includes functions that store and update this information.

The classes described below are only a subset of LASER components. For a full list of all LASER classes, see the [API reference](../reference/laser/generic/index.md).

## LaserFrame and PropertySet

`LaserFrame` can be thought of as a database system for LASER models. It is a class that is used to dynamically manage and allocate data for nodes and agents, and supports both scalar and vector properties.

`PropertySet` stores agent properties in a dictionary-like object. Properties can be dynamically updated and this class is used to help define simulation parameters.

The tutorial [Build SIR models](../tutorials/sir.md) demonstrates how these two classes can be implemented.


## SortedQue

Timing of events is an important function for agent-based models. Management and tracking of values, especially as the number of agents reaches the millions, can be problematic. `SortedQue` is a custom process created for LASER to track and sort these values, and works directly with `LaserFrame` object arrays.
