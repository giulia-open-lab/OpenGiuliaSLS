# Giulia Documentation

This is the documentation for the Giulia repository.
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

> [!TIP]
> This generates the documentation in HTML format, for serving it from a website.
> You can also generate it in PDF format with:
> ```shell
> make -C . simplepdf
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

Simply run (from `/docs`):
```shell
docker build -t giulia-docs .
```
