# Overview of LASER and laser-generic

## 🔹 What is LASER and laser-generic?

- **LASER (Lightweight Agent Spatial modeling for ERadication)** is a framework for building **agent-based infectious disease models**.
  - Populations are represented as a mutable dataframe, where rows are individuals (agents) and columns are properties (e.g., infection state, age, node_id).:contentReference[oaicite:0]{index=0}
- **laser-generic** builds on top of **laser-core**, offering a set of **ready-to-use, generic disease model components** (e.g., SI, SIS, SIR dynamics, births, deaths, vaccination).
  - Distributed as a Python package:
    ```bash
    pip install laser-generic
    ```
:contentReference[oaicite:1]{index=1}

---

## 🔹 Core Principles of LASER

- **Efficient computation**: preallocated memory, fixed-size arrays, sequential array access, and cache-friendly operations:contentReference[oaicite:2]{index=2}
- **Modular design**: users define properties and add modular **components** (step functions) that run each timestep
- **Extensible**: models can be optimized using **NumPy**, **Numba**, or even C/OpenMP for performance:contentReference[oaicite:3]{index=3}
- **Spatial focus**: agents belong to patches (nodes), with migration modules (gravity, radiation, Stouffer’s rank, etc.) for multi-patch models:contentReference[oaicite:4]{index=4}

---

## 🔹 Key Classes and Utilities (from laser-core)

- **LaserFrame** – custom dataframe for populations, with `.add_scalar_property()` and `.add_vector_property()`:contentReference[oaicite:5]{index=5}
- **PropertySet** – dictionary-like structure for managing model properties:contentReference[oaicite:6]{index=6}
- **SortedQueue** – high-performance event queue:contentReference[oaicite:7]{index=7}
- **Demographics utilities** – initialize births, deaths, and population pyramids:contentReference[oaicite:8]{index=8}
- **Migration module** – gravity, radiation, and other migration models:contentReference[oaicite:9]{index=9}
- **Visualization utilities** – outputs and plots from simulations:contentReference[oaicite:10]{index=10}

---

## 🔹 laser-generic: Provided Models and Components

**laser-generic** provides **generic components** for common epidemiological processes:contentReference[oaicite:11]{index=11}:

### Infection & Transmission
- `Infection()` / `Infection_SIS()` – intrahost progression for SI and SIS models
- `Susceptibility()` – manages agent susceptibility
- `Exposure()` – models exposed (latent) state with timers
- `Transmission()` / `TransmissionSIR()` – interhost transmission dynamics
- `Infect_Agents_In_Patch()` / `Infect_Random_Agents()` – stochastic infection events:contentReference[oaicite:12]{index=12}

### Births & Demographics
- `Births()` – demographic process, assigning DOB and node IDs
- `Births_ConstantPop()` – keeps population constant by matching births to deaths
- `Births_ConstantPop_VariableBirthRate()` – constant population but with variable crude birth rates:contentReference[oaicite:13]{index=13}

### Immunization
- `ImmunizationCampaign()` – age-targeted, periodic campaigns
- `RoutineImmunization()` – ongoing routine immunization at target ages
- `immunize_in_age_window()` – helper to immunize within an age band:contentReference[oaicite:14]{index=14}

### Initialization & Seeding
- `seed_infections_in_patch()` / `seed_infections_randomly()` / `seed_infections_randomly_SI()` – seed infections at start
- `set_initial_susceptibility_in_patch()` / `set_initial_susceptibility_randomly()` – initialize susceptibility:contentReference[oaicite:15]{index=15}

### Utilities
- `calc_capacity()` – computes population capacity given births and ticks
- `calc_distances()` – helper for spatial coupling via geocoordinates
- `get_default_parameters()` – returns baseline parameters:contentReference[oaicite:16]{index=16}

---

## 🔹 Example Models and Tutorials

laser-generic comes with **tutorials and validation checks** that implement canonical epidemiological systems:contentReference[oaicite:17]{index=17}:

- SI model (no births, with births, logistic growth):contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}
- SIS model (no births):contentReference[oaicite:20]{index=20}
- SIR model: outbreak size (Kermack–McKendrick), age of infection, intrinsic periodicity:contentReference[oaicite:21]{index=21}:contentReference[oaicite:22]{index=22}:contentReference[oaicite:23]{index=23}
- Critical Community Size (CCS) in SIR:contentReference[oaicite:24]{index=24}
- Spatial coupling: incidence correlation in 2-patch SIR:contentReference[oaicite:25]{index=25}
- Epidemic wave analysis and periodicity:contentReference[oaicite:26]{index=26}:contentReference[oaicite:27]{index=27}

---

## 🔹 How a Model is Built

1. **Define agent properties** (infection state, timers, demographics) via `LaserFrame.add_scalar_property()`
2. **Assemble components** (infection, transmission, births, immunity, etc.)
3. **Configure model** using `Model(scenario, parameters)`:contentReference[oaicite:28]{index=28}
4. **Run simulation** across timesteps
5. **Generate reports** (CSV outputs, plots, population pyramids, outbreak size, etc.)

---

## ✅ Summary

**laser-generic** provides a **library of reusable epidemiological components** (infection states, births, immunization, transmission, seeding), plus **canonical models (SI, SIS, SIR, spatial)** as templates.

It is designed for researchers who want **flexibility and reproducibility** in agent-based disease modeling, while leveraging **Python data science tools** and optionally accelerating with **Numba/GPU**.
