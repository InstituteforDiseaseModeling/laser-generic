# Software overview

LASER is a modeling framework that includes a variety of ways for users to implement the code to model infectious diseases.

<!-- had to delete a bunch since it was framed around laser-core; need to add some more content/context -->


## Design principles

<!-- Can include relevant software principles, or design choices. Included topics should be things that are unique to laser, such that modelers would need to know what this is in order to utilize laser properly (i.e. don't include general modeling principles, assume that the user already knows those). This can also include the high-level features of LASER, what makes it special. -->

The philosophy driving the development of LASER was to create a framework that was flexible, powerful, and fast, able to tackle a variety of complex modeling scenarios without sacrificing performance. But complexity often slows performance, and not every modeling question requires a full suite of model features. To solve this problem, LASER was designed as a set of core components, each with fundamental features that could be added--or not--to build working models. Users can optimize performance by creating models tailored to their research needs, only using components necessary for their modeling question. This building-block framework enables parsimony in model design, but also facilitates the building of powerful models with bespoke, complex dynamics.

LASER's core principles can be summarized as follows:

- **Efficient computation**: preallocated memory, fixed-size arrays, sequential array access, and cache-friendly operations.
- **Modular design**: users define properties and add modular **components** (step functions) that run each timestep.
- **Fast**: models can be progressively optimized using **NumPy**, **Numba**, or even C/OpenMP for performance.
- **Spatial focus**: agents belong to patches (nodes), with migration modules (gravity, radiation, Stouffer’s rank, etc.) for multi-patch models.

## Software architecture

<!-- Framework of how laser works: insert diagram! -->
<!-- should also include explanations of what core is vs generic or other disease models -->

### Input and output files

<!-- All info on the input files and output files. If there are built-in reports, include those. Any type of data requirements should also be included here. Even if it's just that data needs to have a specific structure, include that here.

Even if there are no "required" files, there still needs to be guidelines on formats, basic information needs, example files, etc. Better to provide some guidelines and let users know they're flexible than to say "anything goes" with out any starting point -->


### Software components

Components are modular units of functionality within the simulation, responsible for performing specific updates or computations on the agent population or node-level data. Each component is implemented as a class with an initialization function to set up any required state and a step function to execute the component’s logic during each timestep.

Components vary in their complexity and functionality. Some may be simple counters or used for general accounting purposes, such as tracking individuals in different diseases states, while others will include disease dynamics or even data analysis functionality. LASER enables users to customize which components are used in order to help customize model functions to specific research questions, while providing basic set up examples and ready-to-use model configurations. The following sections provide background on some of the main components; see the API documentation for all class information.

<!-- [Deep dive into components and how they work, how they comprise laser functionality. Each "type" of component will have a topic section as needed]

Make it clear that this is not a comprehensive list, but a call-out for the various functions the user can play with (link to API docs for full listing of laser functions)


Need to make sure we explain all of the relevant/important parts! Eg, the classes used in the SIR tutorial should be all explained. -->
