FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    USE_PYGEOS=0 \
    SPATIALITE_LIBRARY_PATH='mod_spatialite.so' \
    SHELL=bash \
    TINI_VERSION=v0.19.0

# tini init
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# System deps + python tooling
RUN apt update && \
    apt install -y --no-install-recommends \
      python3-full python3-dev python3-venv python3-pip \
      build-essential postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create & activate venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy app code
WORKDIR /opt/app

# Copy requirements + constraints into /conf
RUN mkdir -p /conf
COPY requirements.txt constraints.txt /conf/

# Install Python deps using the constraint
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      -r /conf/requirements.txt \
      -c /conf/constraints.txt

COPY . .

# Expose & run
EXPOSE 8501
ENTRYPOINT ["/tini", "--"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# Add health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Create a simple health check endpoint
RUN echo 'import streamlit as st\nst.set_page_config(initial_sidebar_state="collapsed")\nst.title("Health Check")\nst.success("OK")' > health_check.py