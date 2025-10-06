🔹 What is LASER and laser-generic?
===================================

- **LASER (Lightweight Agent Spatial modeling for ERadication)** is a framework for building
  **agent-based infectious disease models** with an emphasis on spatial modeling and heterogeneity
  and efficient computation at scale.

  Populations are represented as a mutable dataframe, where rows are individuals (agents)
  and columns are properties (e.g., infection state, age, node_id).

- **laser-generic** builds on top of **laser-core**, offering a set of
  **ready-to-use, generic disease model components** (e.g., SI, SIS, SIR dynamics, births, deaths, vaccination).

---

🔹 Core Principles of LASER
===========================

- **Efficient computation**: preallocated memory, fixed-size arrays, sequential array access, and cache-friendly operations.
- **Modular design**: users define properties and add modular **components** (step functions) that run each timestep.
- **Fast**: models can be progressively optimized using **NumPy**, **Numba**, or even C/OpenMP for performance.
- **Spatial focus**: agents belong to patches (nodes), with migration modules (gravity, radiation, Stouffer’s rank, etc.) for multi-patch models.

---

🔹 Key Classes and Utilities (from laser-core)
=============================================

- **LaserFrame** – custom dataframe for populations, with ``.add_scalar_property()`` and ``.add_vector_property()``.
- **PropertySet** – dictionary-like structure for managing model properties.
- **SortedQueue** – high-performance event queue.
- **Demographics utilities** – initialize births, deaths, and population pyramids.
- **Migration module** – gravity, radiation, and other migration models.
- **Visualization utilities** – outputs and plots from simulations.

---

🔹 laser-generic: Provided Models and Components
===============================================

**laser-generic** provides **generic components** built atop laser-core for common epidemiological processes:

**Infection & Transmission**

- ``Infection()`` / ``Infection_SIS()`` – intrahost progression for SI and SIS models.
- ``Susceptibility()`` – manages agent susceptibility.
- ``Exposure()`` – models exposed (latent) state with timers.
- ``Transmission()`` / ``TransmissionSIR()`` – interhost transmission dynamics.
- ``Infect_Agents_In_Patch()`` / ``Infect_Random_Agents()`` – stochastic infection events.

**Births & Demographics**

- ``Births()`` – demographic process, assigning DOB and node IDs.
- ``Births_ConstantPop()`` – keeps population constant by matching births to deaths.
- ``Births_ConstantPop_VariableBirthRate()`` – constant population but with variable crude birth rates.

**Immunization**

- ``ImmunizationCampaign()`` – age-targeted, periodic campaigns.
- ``RoutineImmunization()`` – ongoing routine immunization at target ages.
- ``immunize_in_age_window()`` – helper to immunize within an age band.

**Initialization & Seeding**

- ``seed_infections_in_patch()`` / ``seed_infections_randomly()`` / ``seed_infections_randomly_SI()`` – seed infections at start.
- ``set_initial_susceptibility_in_patch()`` / ``set_initial_susceptibility_randomly()`` – initialize susceptibility.

**Utilities**

- ``calc_capacity()`` – computes population capacity given births and ticks.
- ``calc_distances()`` – helper for spatial coupling via geocoordinates.
- ``get_default_parameters()`` – returns baseline parameters.

---

🔹 Example Models and Tutorials
===============================

laser-generic comes with **tutorials and validation checks** that implement canonical epidemiological systems:

- SI model (no births, with births, logistic growth)
- SIS model (no births)
- SIR model: outbreak size (Kermack–McKendrick), age of infection, intrinsic periodicity
- Critical Community Size (CCS) in SIR
- Spatial coupling: incidence correlation in 2-patch SIR
- Epidemic wave analysis and periodicity

---

✅ Summary
==========

**laser-generic** provides a **library of reusable epidemiological components** (infection states, births,
immunization, transmission, seeding), plus **canonical models (SI, SIS, SIR, spatial)** as templates.

It is designed for researchers who want **flexibility and reproducibility** in agent-based disease modeling,
while leveraging **Python data science tools** and optionally accelerating with **Numba/GPU**.
