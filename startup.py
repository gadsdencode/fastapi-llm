#!/usr/bin/env python3
"""
Azure App Service startup script for FastAPI LLM server
Simplified for Azure compatibility with proper error handling
"""
import os
import sys
import logging

# Add the current directory to Python path to fix import issues
sys.path.insert(0, '/home/site/wwwroot')
sys.path.insert(0, '/home/site/wwwroot/app')

# Configure logging for Azure App Service
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Start the FastAPI application with Azure-optimized configuration"""
    try:
        # Log startup information
        logger.info("Starting Azure App Service FastAPI application")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"PORT environment variable: {os.getenv('PORT', '8000')}")
        
        # Azure App Service specific configuration
        port = int(os.getenv("PORT", 8000))
        workers = 1  # Start with single worker for Azure
        
        # Azure tier detection
        azure_sku = os.getenv("WEBSITE_SKU", "Free")
        logger.info(f"Azure App Service tier: {azure_sku}")
        
        # Import uvicorn here to avoid early import issues
        import uvicorn
        
        # Try to import the FastAPI app
        try:
            from app.main import app
            logger.info("Successfully imported app from app.main")
        except ImportError as e:
            logger.error(f"Failed to import app.main: {e}")
            # Try alternative import paths
            try:
                import main
                app = main.app
                logger.info("Successfully imported app from main module")
            except ImportError as e2:
                logger.error(f"Failed to import main: {e2}")
                sys.exit(1)
        
        # Configure uvicorn for Azure App Service
        uvicorn_config = {
            "app": app,
            "host": "0.0.0.0",
            "port": port,
            "workers": workers,
            "log_level": "info",
            "access_log": True,
            "use_colors": False,  # Disable colors for Azure logs
            "timeout_keep_alive": 30,
            "timeout_graceful_shutdown": 10
        }
        
        logger.info(f"Starting uvicorn server on 0.0.0.0:{port}")
        uvicorn.run(**uvicorn_config)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 