# Disease transmission



## Transmission modes

The LASER framework includes modules for canonical transmission dynamics, including SI, SIS, SIR, SIRS, SEIR, and SEIRS.
<!-- add content on the disease model components: SI, SIR, SIER, etc. -->

To build and explore model dynamics, see the [Tutorials](../tutorials/index.md) section.

## Immunization

Immunity can be introduced into the agent population through several component types.

Routine immunization (RI) can be implemented, such that at specified time steps susceptible agents whose age falls within a specified age range will be selected through a binomial draw to be immunized. Routine immunization events can be set up as recurring events that will continuously sample agents in the specified age range at specified time steps to provide immunization coverage.

Immunization campaigns can be implemented, such that agents within an age band can be targeted for specific immunization events.
