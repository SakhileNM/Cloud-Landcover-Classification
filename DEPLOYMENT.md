# Oracle Cloud Deployment

## Instance Configuration
- **Shape**: VM.Standard.A1.Flex
- **OCPUs**: 4
- **Memory**: 24GB
- **Storage**: 50GB
- **OS**: Ubuntu 22.04 LTS

## Deployment Steps

1. Create Oracle Cloud account
2. Launch A1.Flex instance with above configuration
3. Configure VCN with ports 22 (SSH) and 8501 (App) open
4. Connect via SSH and run:
   ```bash
   git clone https://github.com/SakhileNM/Cloud-Landcover-Classification.git
   cd Cloud-Landcover-Classification
   docker build -t landcover-app .
   docker run -d -p 8501:8501 --name landcover-container landcover-app
