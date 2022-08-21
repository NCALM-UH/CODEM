FROM continuumio/miniconda3

# Keep things clean
WORKDIR /code

# Create and activate the Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "source activate modem" >> ~/.bashrc

# Patch a problem
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

# Some web copy/paste to automatically activate the conda environment
ENV CONDA_EXE /opt/conda/bin/conda
ENV CONDA_PREFIX /opt/conda/envs/modem
ENV CONDA_PYTHON_EXE /opt/conda/bin/python
ENV CONDA_PROMPT_MODIFIER (modem)
ENV CONDA_DEFAULT_ENV modem
ENV PATH /opt/conda/envs/modem/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install the Python CLI
COPY src ./src
COPY readme.md .
COPY setup.py .
RUN pip install .
