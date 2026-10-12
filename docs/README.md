# Giulia Documentation

This is the documentation for the Giulia repository — a data-driven, event-driven
system-level radio network simulator.

## Documentation map

| Topic | Where | What it covers |
|---|---|---|
| **Energy efficiency** | [`energy-efficiency/`](energy-efficiency/README.md) | RAN power-saving features: cell roles, [pairing](energy-efficiency/pairing.md), the [RRC layer](energy-efficiency/rrc-layer.md), the surrounding [RRM chain](energy-efficiency/rrm.md), and the [carrier-shutdown](energy-efficiency/carrier-shutdown.md) algorithm — with what it actually saves and where it does not. Start at the [overview](energy-efficiency/overview.md). |
| **Saveable & monitoring** | [`output-saveable/`](output-saveable/README.md) | How a class exposes its results, how those results are written to Parquet (and read back), the catalog of available variables per preset, live KPI monitoring, and the [per-context registry](output-saveable/saveable-concurrency.md) that keeps parallel simulations apart. The Sphinx-wired "objects" page is [`source/saveable-objects.rst`](source/saveable-objects.rst). |
| **Dedalo** | [`dedalo/`](dedalo/README.md) | The configuration interface to Giulia: how an external script, optimizer or agent modifies a live simulation. Covers the [API contract](dedalo/api.md) and its lifecycle, [`df_ep` consistency](dedalo/df-ep-consistency.md) (derived fields, the checklist, and what a naive write costs), [reading results back](dedalo/kpis.md) (through `Saveable`, not through Dedalo), and [worked examples](dedalo/examples.md). Start at the [overview](dedalo/overview.md). |

> **Markdown vs. Sphinx.** Both are part of the same site. The pages under
> [`source/`](source/) are reStructuredText; the top-level Markdown pages are pulled
> in by thin `.rst` wrappers that `include` them through
> [MyST](https://myst-parser.readthedocs.io/) (for example
> [`source/architecture.rst`](source/architecture.rst) renders
> [`architecture.md`](architecture.md)). The Markdown is written so it also reads well
> directly on a Git host or in an editor — Mermaid diagrams and relative links render
> there too.
>
> The `energy-efficiency/`, `output-saveable/` and `dedalo/` sets are the exception:
> they are Markdown-only and deliberately **not** in the Sphinx `toctree`, because they
> are deep-dive developer notes that link heavily into source files by line number. Read
> them on the Git host.
>
> If you add a page, remember to add it to the `toctree` in
> [`source/index.rst`](source/index.rst) — Sphinx warns about, but still builds,
> orphan documents, so it is easy to miss.

## Building the Sphinx documentation

In order to build the documentation, you need to set up the `docs` environment:

```bash
conda env create -n giulia-docs --file env/environment-docs.yml
```

Then, activate it:

```bash
conda activate giulia-docs
```

Finally, build the documentation:

```bash
make -C docs html
```

> [!NOTE]
> The docs environment is independent of the simulator's: you do **not** need the GPU
> stack (Sionna / PyTorch / CUDA) to build the documentation. If you prefer pip over
> Conda, `pip install -r docs/requirements.txt` installs the same set. The two files
> pin the same packages and must be kept in sync.

> [!TIP]
> This generates the documentation in HTML format, for serving it from a website.
> You can also generate it in PDF format with:
> ```shell
> make -C docs simplepdf
> ```

> [!IMPORTANT]
> Most browsers have an automatic cache for the documentation, so if you don't see the changes you made, try to clear
> the cache or open the documentation in a private window.
> 
> In Chrome:
> Press `F12` to open the Developer Tools, then right-click on the refresh button and select `Empty Cache and Hard Reload`.
> 
> In Firefox:
> Press `Ctrl+F5` to perform a hard reload.

The documentation will be available at `docs/build/html/index.html`.

We provide a Docker Compose file to start a container with the documentation server.
The built files are attached using a volume, so you don't need to restart the container every time you build the
documentation, simply build it and refresh the page.

To use it, you will need [Docker Compose](https://docs.docker.com/compose/). Then, run:

```bash
cd docs
docker compose -f docker-compose.dev.yml up
```

Now you can access the documentation at http://localhost:1080.

> [!TIP]
> `localhost` may not be the correct hostname. If you are running in a remote server, you should replace `localhost`
> with the IP of that server.
> 
> Example: `http://158.42.160.122:1080`

Whenever you update the documentation, you still have to run `make -C docs html` again, to refresh the generated source.

To stop the container press `Ctrl+C`, and to perform a cleanup run:

```bash
docker compose down
```

## Packaging with Docker

If you want to share the documentation with someone else, you can package it with Docker.
[`Dockerfile`](Dockerfile) runs the Sphinx build **inside** the image, so you do not need to build
the HTML first.

Simply run (from `/docs`):
```shell
docker build -t giulia-docs .
```

Then serve it:
```shell
docker run --rm -p 8080:80 giulia-docs
```

The equivalent Compose files are:

| File | What it does | Port |
|---|---|---|
| [`docker-compose.yml`](docker-compose.yml) | Builds the image from `Dockerfile` and serves it | 8080 |
| [`docker-compose.dev.yml`](docker-compose.dev.yml) | Serves the already-built `build/html` through a volume | 1080 |
| [`docker-compose.prod.yml`](docker-compose.prod.yml) | Pulls the published `iteamupv/giulia:latest`; exposes no port, expects a reverse proxy | — |

The published image and the GitHub Pages deployment are both produced by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) on pushes to `main`.
