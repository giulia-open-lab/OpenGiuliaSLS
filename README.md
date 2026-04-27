# Giulia

[![Documentation](https://a11ybadges.com/badge?text=Documentation&logo=book-open)](http://158.42.160.122/)
[![License](https://a11ybadges.com/badge?text=License&logo=shield)](https://github.com/david-lopez-perez/Giulia/blob/main/LICENSE)

[![Matrix](https://a11ybadges.com/badge?logo=matrix)](https://matrix.to/#/!hbiCQOdYflhWRcHwyT:matrix.org?via=matrix.org)
[![Docker Hub](https://a11ybadges.com/badge?text=DockerHub&logo=docker&badgeColor=#4287f5)](https://hub.docker.com/repository/docker/iteamupv/giulia/general)

**Data-driven system level simulator**

This repository hosts a one-of-a-kind system-level simulator developed in Python. It is aimed at encompassing both
cellular and Wi-Fi capabilities and is based on an event-driven philosophy.

To run the simulator, use the 'main.py' function and select the desired use case or scenario for simulation.

---

![Attention](https://a11ybadges.com/badge?text=Attention&logo=alert-triangle&badgeColor=rgb(235,180,28))

**Please note that the simulator is currently under construction. We appreciate your patience.**

---

### Dependencies
This simulator is using Sionna by NVIDIA which requires TensorFlow 2.10-2.15 and Python 3.8-3.11. 

Other related packages: NumPy; Pandas; SciPy; Simulus; Shapely; GeoPandas; MatplotLib; scikit-learn; itur;

See [the official documentation](https://giulia-docs.arnaumora.com/setup-environment.html) to know more about the
installation process.

### Running Giulia

The easiest way to run Giulia is to use configuration files.
Those allow you to easily select how to run Giulia, and which outputs you want to get.

We provide an example inputs file with the default values for you to configure: [`config.example.yml`](config.example.yml)

Then, run the following command:

```shell
python main.py
```

Please note that this command expects to be inside a valid environment with all the required dependencies installed.

Giulia will by default try to find this file on the root of the repo, with the name: `config.yml`
If you want to set a custom file, specify its full path in the command, for example:

```shell
python main.py config.example.yml
```

### Running examples directly (batches and scripts)

If instead you are interested in reproducing batch runs or scripts directly — for example to drive several scenarios from Python without going through the YAML config — run the example script directly:

```shell
python examples/giulia_EV.py
```

In this mode the YAML config is **not** consulted. Configuration comes entirely from command-line flags and the parser defaults defined in [`examples/giulia_EV.py`](examples/giulia_EV.py). Running it with no flags starts a simulation with the following defaults:

- `preset`: `GiuliaStd`
- `scenario_model`: `ITU_R_M2135_UMa`
- `ue_playground_model`: `None` (uses the scenario's default playground)
- `ue_distribution`: `uniform`
- `ue_mobility`: `None` (static UEs)
- `link_direction`: `downlink`
- `wraparound`: `None`
- `snapshots`: `2`
- `enable_GPU`: `True`
- `regression`: `False`
- `log_level`: `0`
- `save_results`: `1`
- `plots`: `True`

To run a different scenario or override any other default, pass the corresponding flag, e.g.:

```shell
python examples/giulia_EV.py --scenario_model 3GPPTR38_901_5G --ue_distribution uniform --snapshots 10
```

For driving several scenarios in a row, see [`examples/giulia_EV_batch.py`](examples/giulia_EV_batch.py), which imports `giulia_EV` and calls its `main(args)` once per scenario.

#### Multilayer scenario naming convention

In the multilayer scenario names under `3GPPTR38_901_*_multilayer`, the underscores are meaningful:

- `4G5G` (no underscore between layers) — the listed layers are **colocated** at the same sites.
- `4G_5G` (underscore between layers) — the listed layers are **non-colocated**; in this case the 5G layer is deployed on the UMi scenario.

The same logic applies to the 6G variants:

- `3GPPTR38_901_4G5G_multilayer` — 4G and 5G colocated.
- `3GPPTR38_901_4G_5G_multilayer` — 4G and 5G non-colocated, 5G on UMi.
- `3GPPTR38_901_4G_5G6G_multilayer` — 4G non-colocated from the 5G+6G pair, which is itself colocated.
- `3GPPTR38_901_4G_5G_6G_multilayer` — 4G, 5G and 6G all non-colocated, with 5G and 6G on their respective UMi scenarios.

### Running tests

Giulia is tested by running different models and distributions, and comparing them to some precomputed ones.

Know more in the tests guide: [access](./docs/testing.md)
