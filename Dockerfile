FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    USE_PYGEOS=0 \
    SHELL=bash

# System deps + python tooling
RUN apt update && \
    apt install -y --no-install-recommends \
      python3-full python3-dev python3-venv python3-pip \
      build-essential postgresql-client curl wget \
      libgdal-dev gdal-bin python3-gdal \
    && rm -rf /var/lib/apt/lists/*

# Create & activate venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy app code
WORKDIR /opt/app

# Copy requirements
COPY requirements.txt ./

# Install Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose & run
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
