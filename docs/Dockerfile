FROM alpine AS build

# Install make
RUN apk add --update --no-cache make

# Install python/pip
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 python3-dev py3-pip && ln -sf python3 /usr/bin/python

# Install GCC for PDF conversion (not need on Docker, but required as dependency)
RUN apk add build-base

# Create a new virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN . /venv/bin/activate

# Install the required python packages
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# Copy the documentation
COPY ./ /docs/
WORKDIR /docs/

# Build the HTML files
RUN make -C /docs/ html

FROM httpd:2.4-alpine

# Copy the built HTML files to the web server
COPY --from=build /docs/build/html/ /usr/localapache2/htdocs/
