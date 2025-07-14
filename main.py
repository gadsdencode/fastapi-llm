"""
Simple main.py file for Azure App Service compatibility
This file helps Azure App Service find the FastAPI application
"""
try:
    from app.main import app
except ImportError:
    # Fallback for Azure App Service path issues
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
    from main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)