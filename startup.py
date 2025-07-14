#!/usr/bin/env python3
"""
Azure App Service startup script for FastAPI LLM server
Optimized for Azure Web App deployment with proper logging and error handling
"""
import os
import sys
import logging
import uvicorn

# Configure logging for Azure App Service
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def get_azure_optimized_config():
    """Get Azure-optimized uvicorn configuration based on App Service tier"""
    port = int(os.getenv("PORT", 8000))
    
    # Detect Azure App Service tier for optimal configuration
    azure_sku = os.getenv("WEBSITE_SKU", "Free")
    workers = 1  # Azure App Service works best with single worker
    
    logger.info(f"Azure App Service tier detected: {azure_sku}")
    logger.info(f"Starting FastAPI LLM server on port {port}")
    
    config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": port,
        "workers": workers,
        "timeout_keep_alive": 300,
        "log_level": "info",
        "access_log": True,
        "reload": False,
        "use_colors": False,  # Disable colors for Azure logs
    }
    
    # Azure-specific optimizations
    if azure_sku in ["Free", "Shared"]:
        # Conservative settings for lower tiers
        config.update({
            "timeout_keep_alive": 120,
            "limit_concurrency": 10,
            "limit_max_requests": 100,
        })
        logger.info("Applied Free/Shared tier optimizations")
    elif azure_sku in ["Basic"]:
        # Basic tier optimizations
        config.update({
            "timeout_keep_alive": 200,
            "limit_concurrency": 50,
            "limit_max_requests": 500,
        })
        logger.info("Applied Basic tier optimizations")
    else:
        # Standard/Premium tier optimizations
        config.update({
            "timeout_keep_alive": 300,
            "limit_concurrency": 100,
            "limit_max_requests": 1000,
        })
        logger.info("Applied Standard/Premium tier optimizations")
    
    # Use standard asyncio for better Azure compatibility
    config["loop"] = "asyncio"
    config["http"] = "h11"  # Use h11 for better Azure compatibility
    
    return config


def main():
    """Main startup function for Azure App Service"""
    try:
        logger.info("Starting Azure App Service FastAPI LLM server...")
        
        # Set Azure deployment flag
        os.environ["AZURE_DEPLOYMENT"] = "true"
        
        # Get optimized configuration
        config = get_azure_optimized_config()
        
        # Start the server
        uvicorn.run(**config)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 