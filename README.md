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

### Running tests

Giulia is tested by running different models and distributions, and comparing them to some precomputed ones.

Know more in the tests guide: [access](./TESTS.md)
