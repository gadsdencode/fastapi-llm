# Azure Web App Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the FastAPI LLM Inference Server to Azure Web App (App Service) with all optimizations and compatibility improvements implemented.

## Azure Web App Compatibility Features Implemented

### ✅ Azure-Specific Optimizations

1. **Startup Configuration**
   - `startup.py` - Azure-optimized startup script with tier detection
   - `azure.yaml` - Azure App Service configuration file
   - Automatic tier detection using `WEBSITE_SKU` environment variable

2. **Resource Optimization by Tier**
   - **Free/Shared Tier**: 1 thread, 512 context, conservative limits
   - **Basic Tier**: 2 threads, 1024 context, moderate limits  
   - **Standard/Premium Tier**: 4 threads, 2048 context, full performance

3. **Model Loading Enhancements**
   - 5-minute timeout for Azure App Service constraints
   - Lazy loading by default (`LOAD_MODEL_ON_STARTUP=false`)
   - Azure-specific error handling and warnings

4. **Dockerfile Optimizations**
   - Azure-compatible user permissions
   - Optimized health checks for Azure
   - Azure-specific environment variables

## Deployment Methods

### Method 1: GitHub Actions (Recommended)

The repository includes a pre-configured GitHub Actions workflow for automatic deployment.

#### Prerequisites
- Azure subscription
- Azure Web App created
- GitHub repository with deployment secrets configured

#### Setup Steps

1. **Create Azure Web App**
   ```bash
   # Using Azure CLI
   az webapp create \
     --resource-group myResourceGroup \
     --plan myAppServicePlan \
     --name FastAPI-LLM-Server \
     --runtime "PYTHON|3.11" \
     --deployment-container-image-name python:3.11-slim
   ```

2. **Configure GitHub Secrets**
   Add these secrets to your GitHub repository:
   - `AZUREAPPSERVICE_CLIENTID_*`
   - `AZUREAPPSERVICE_TENANTID_*`
   - `AZUREAPPSERVICE_SUBSCRIPTIONID_*`

3. **Deploy via GitHub Actions**
   - Push to `clean-dev` branch
   - GitHub Actions will automatically build and deploy

### Method 2: Azure CLI Deployment

#### Step 1: Prepare Local Environment
```bash
# Clone repository
git clone <repository-url>
cd fastapi-llm

# Install Azure CLI (if not installed)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

#### Step 2: Login to Azure
```bash
az login
az account set --subscription "your-subscription-id"
```

#### Step 3: Create Resource Group and App Service Plan
```bash
# Create resource group
az group create --name fastapi-llm-rg --location "East US"

# Create App Service plan (choose appropriate tier)
az appservice plan create \
  --name fastapi-llm-plan \
  --resource-group fastapi-llm-rg \
  --sku B1 \
  --is-linux
```

#### Step 4: Create Web App
```bash
az webapp create \
  --resource-group fastapi-llm-rg \
  --plan fastapi-llm-plan \
  --name your-unique-app-name \
  --runtime "PYTHON|3.11"
```

#### Step 5: Configure Application Settings
```bash
# Set required environment variables
az webapp config appsettings set \
  --resource-group fastapi-llm-rg \
  --name your-unique-app-name \
  --settings \
    AZURE_DEPLOYMENT=true \
    LOAD_MODEL_ON_STARTUP=false \
    MODEL_TYPE=gguf \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
```

#### Step 6: Deploy Application
```bash
# Deploy from local directory
az webapp deploy \
  --resource-group fastapi-llm-rg \
  --name your-unique-app-name \
  --src-path . \
  --type zip
```

### Method 3: Docker Container Deployment

#### Step 1: Build and Push Docker Image
```bash
# Build Azure-optimized image
docker build -t your-registry/fastapi-llm:azure .

# Push to Azure Container Registry or Docker Hub
docker push your-registry/fastapi-llm:azure
```

#### Step 2: Create Web App from Container
```bash
az webapp create \
  --resource-group fastapi-llm-rg \
  --plan fastapi-llm-plan \
  --name your-unique-app-name \
  --deployment-container-image-name your-registry/fastapi-llm:azure
```

## Environment Variables Configuration

### Required Variables
| Variable | Value | Description |
|----------|--------|-------------|
| `AZURE_DEPLOYMENT` | `true` | Enables Azure optimizations |
| `LOAD_MODEL_ON_STARTUP` | `false` | Prevents startup timeout |
| `MODEL_TYPE` | `gguf` | Recommended model type |
| `PYTHONPATH` | `/app` | Python path for imports |

### Optional Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | Auto-detected | Specific model to load |
| `OMP_NUM_THREADS` | Auto-detected | CPU threads (tier-based) |
| `MKL_NUM_THREADS` | Auto-detected | Math library threads |
| `LLM_MAX_CONTEXT` | Auto-detected | Context size (tier-based) |

### Azure-Specific Variables (Auto-detected)
| Variable | Description |
|----------|-------------|
| `WEBSITE_SKU` | Azure App Service tier (Free, Basic, Standard, Premium) |
| `WEBSITE_SITE_NAME` | App name |
| `WEBSITE_HOSTNAME` | App URL |

## Tier-Specific Recommendations

### Free Tier (F1)
- **Limitations**: 60 minutes/day, no custom domains
- **Recommended Models**: Lightweight models < 1GB
- **Configuration**: 1 thread, 512 context
- **Use Case**: Development and testing only

### Basic Tier (B1-B3)
- **Limitations**: No auto-scaling, basic SSL
- **Recommended Models**: Small to medium models < 3GB
- **Configuration**: 2 threads, 1024 context  
- **Use Case**: Small production workloads

### Standard Tier (S1-S3)
- **Features**: Auto-scaling, staging slots, backups
- **Recommended Models**: Medium to large models < 8GB
- **Configuration**: 4 threads, 2048 context
- **Use Case**: Production workloads

### Premium Tier (P1V2-P3V2)
- **Features**: Enhanced performance, VNet integration
- **Recommended Models**: Large models, multiple models
- **Configuration**: Optimized threading, full context
- **Use Case**: High-performance production

## Model Loading Strategy

### Default Behavior (Azure Optimized)
1. **Startup**: Server starts without loading models (fast startup)
2. **First Request**: Model loads on-demand with timeout protection
3. **Caching**: Models cached in `/app/models` directory
4. **Fallback**: Smaller models recommended for lower tiers

### Recommended Models by Tier

#### Free/Basic Tiers
```bash
# Lightweight models
MODEL_NAME=microsoft/DialoGPT-small
MODEL_NAME=distilbert-base-uncased
```

#### Standard/Premium Tiers
```bash
# Full-featured models
MODEL_NAME=TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF
MODEL_NAME=v8karlo/UNCENSORED-TinyDolphin-3x-MoE-Q4_K_M-GGUF
```

## Health Monitoring

### Health Check Endpoints
- **Primary**: `/ping` - Simple health check
- **Detailed**: `/api/v1/health` - Comprehensive health information

### Azure Application Insights Integration
```bash
# Enable Application Insights
az webapp config appsettings set \
  --name your-app-name \
  --resource-group your-rg \
  --settings APPLICATIONINSIGHTS_CONNECTION_STRING="your-connection-string"
```

## Troubleshooting

### Common Issues

1. **Startup Timeout**
   - **Cause**: Model loading taking too long
   - **Solution**: Ensure `LOAD_MODEL_ON_STARTUP=false`
   - **Check**: Azure startup logs in App Service logs

2. **Memory Limitations**
   - **Cause**: Model too large for tier
   - **Solution**: Use smaller model or upgrade tier
   - **Monitor**: Memory usage in Azure portal

3. **Import Errors**
   - **Cause**: Missing dependencies or wrong Python path
   - **Solution**: Verify `PYTHONPATH=/app` and requirements.txt
   - **Debug**: Check deployment logs

### Debugging Commands
```bash
# Stream logs
az webapp log tail --name your-app-name --resource-group your-rg

# Download logs
az webapp log download --name your-app-name --resource-group your-rg

# Check app settings
az webapp config appsettings list --name your-app-name --resource-group your-rg
```

## Performance Optimization

### Scaling Configuration
```bash
# Enable auto-scaling (Standard tier and above)
az monitor autoscale create \
  --resource-group your-rg \
  --resource your-app-name \
  --resource-type Microsoft.Web/serverfarms \
  --name autoscale-rules \
  --min-count 1 \
  --max-count 3 \
  --count 1
```

### Custom Domain and SSL
```bash
# Add custom domain (Standard tier and above)
az webapp config hostname add \
  --webapp-name your-app-name \
  --resource-group your-rg \
  --hostname yourdomain.com
```

## Cost Optimization

### Tier Selection Guide
- **Development**: Free tier for testing
- **Small Production**: Basic B1 ($~13/month)
- **Medium Production**: Standard S1 ($~56/month)  
- **High Performance**: Premium P1V2 ($~73/month)

### Resource Management
- Use deployment slots for staging (Standard+)
- Enable auto-scaling to handle traffic spikes
- Monitor usage with Azure Cost Management

## Security Best Practices

### Application Security
```bash
# Enable HTTPS only
az webapp update --name your-app-name --resource-group your-rg --https-only

# Configure authentication (if needed)
az webapp auth update \
  --name your-app-name \
  --resource-group your-rg \
  --enabled true \
  --action LoginWithAzureActiveDirectory
```

### API Key Configuration
```bash
# Set API key for production
az webapp config appsettings set \
  --name your-app-name \
  --resource-group your-rg \
  --settings API_KEY="your-secure-api-key"
```

## Next Steps

1. **Deploy**: Follow one of the deployment methods above
2. **Test**: Use the health check endpoints to verify deployment
3. **Monitor**: Set up Application Insights for monitoring
4. **Scale**: Adjust tier based on usage requirements
5. **Secure**: Configure authentication and API keys for production

## Support and Resources

- **Azure Documentation**: [Python on Azure App Service](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python)
- **FastAPI Documentation**: [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- **Azure Support**: Available through Azure Portal
- **Repository Issues**: Create GitHub issues for application-specific problems 