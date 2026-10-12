FROM continuumio/miniconda3:23.5.2-0-alpine AS giulia-env
ARG INCLUDE_SOURCE

# Set the working dir
WORKDIR /giulia

# copy environment file and install it
COPY ./env/environment.yml environment.yml
RUN conda env create -f environment.yml -n giulia
RUN echo "source activate giulia" > ~/.bashrc
RUN echo "source activate giulia" > ~/.profile
ENV PATH=/opt/conda/envs/giulia/bin:$PATH

# Create a symbolic link for LLVM 14
RUN ln -s /opt/conda/envs/giulia/lib/libLLVM-14.so /opt/conda/envs/giulia/lib/libLLVM.so

# Copy the source files
COPY ./giulia /giulia/giulia/
# Copy the examples dir
COPY ./examples /giulia/examples/
# Copy the tests dir
COPY ./tests /giulia/tests/

# Copy the data dir
COPY ./data /giulia/data/

# If INCLUDE_SOURCE is false, remove the /giulia dir
RUN if [[ -z "$INCLUDE_SOURCE" ]] ; then echo "Won't include source code" && rm -rf /giulia ; else echo "Source code will be bundled into image" ; fi
