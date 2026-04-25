# Giulia user documentation

Giulia is a system-level simulator for cellular and Wi-Fi networks, written in Python and built around an event-driven core. It is aimed at evaluating end-to-end behaviour of a radio access network — base-station deployments, antenna patterns, propagation channels, cell association, scheduling, link-level abstractions, and energy consumption — under a range of standardised 3GPP and ITU-R scenarios as well as custom data-driven scenarios.

This site is the user-facing documentation. It is intended for people who want to install Giulia, configure it through its YAML config file, and run simulations. For information about contributing to the simulator's source code, see the project's developer documentation under `docs/source/`.

## Table of contents

- [Architecture](architecture.md) — how the simulator is organised, the main pipeline stages, and the role of each top-level package under `giulia/`.
- [Environment setup](environment-setup.md) — prerequisites, supported platforms, the installer script, and how to verify the GPU and build this documentation.
- [Models](models.md) — supported scenario / propagation models, physical-layer models, and the underlying scientific stack.
- [Running Giulia](running-giulia.md) — how to use the YAML config, what each `run.example` value dispatches to, where outputs land, and how to customise per-scenario base-station parameters.
- [Channel generation](channel-generation.md) — the step-by-step procedure used to build a UE-to-cell channel, with the standards each step is based on.
- [Testing](testing.md) — how to run the regression test suite and generate test reports.
