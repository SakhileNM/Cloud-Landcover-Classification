FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    USE_PYGEOS=0

# Install system dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
      libgdal-dev gdal-bin \
      postgresql-client curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# Copy requirements first for better caching
COPY requirements.txt .

# Install GDAL first with compatible version, then other packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "gdal==3.4.1.*" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
