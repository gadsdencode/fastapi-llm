"""Configuration management utilities for chat templates and model settings"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import lru_cache

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages chat template loading, caching, and resolution"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize template manager with optional custom config path"""
        if config_path is None:
            config_path = Path(__file__).parent / "templates.yaml"
        
        self.config_path = config_path
        self._templates_cache: Optional[Dict[str, Any]] = None
    
    @property
    def templates(self) -> Dict[str, Any]:
        """Lazy-loaded templates with caching"""
        if self._templates_cache is None:
            self._templates_cache = self._load_templates()
        return self._templates_cache
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from YAML configuration"""
        try:
            if not self.config_path.exists():
                logger.error(f"Template configuration not found: {self.config_path}")
                return self._get_fallback_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded template configuration from {self.config_path}")
                return config
                
        except Exception as e:
            logger.error(f"Failed to load template configuration: {e}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Provide minimal fallback configuration if YAML loading fails"""
        logger.warning("Using fallback template configuration")
        return {
            "templates": {
                "chatml": {
                    "patterns": ["default"],
                    "format": {
                        "system_start": "<|im_start|>system\n",
                        "system_end": "<|im_end|>\n",
                        "user_start": "<|im_start|>user\n",
                        "user_end": "<|im_end|>\n",
                        "assistant_start": "<|im_start|>assistant\n",
                        "assistant_end": "<|im_end|>\n"
                    },
                    "stop_tokens": ["<|im_end|>", "<|im_start|>"],
                    "default_system_message": "You are a helpful AI assistant."
                }
            },
            "settings": {
                "fallback_template": "chatml",
                "default_system_message": "You are a helpful AI assistant."
            }
        }
    
    def get_template_for_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get template configuration for a specific model
        
        Args:
            model_name: Name of the model to get template for
            
        Returns:
            Template configuration dict or None if not found
        """
        if not model_name:
            return self._get_fallback_template()
        
        model_lower = model_name.lower()
        templates = self.templates.get("templates", {})
        
        # Check resolution order if specified
        resolution_order = self.templates.get("resolution_order", templates.keys())
        
        for template_name in resolution_order:
            if template_name not in templates:
                continue
                
            template_config = templates[template_name]
            patterns = template_config.get("patterns", [])
            
            # Check if any pattern matches the model name
            for pattern in patterns:
                if pattern.lower() in model_lower:
                    logger.info(f"Selected template '{template_name}' for model '{model_name}' (matched pattern: '{pattern}')")
                    return template_config
        
        # Fallback to default template
        logger.info(f"No specific template found for model '{model_name}', using fallback")
        return self._get_fallback_template()
    
    def _get_fallback_template(self) -> Dict[str, Any]:
        """Get the fallback template configuration"""
        fallback_name = self.templates.get("settings", {}).get("fallback_template", "chatml")
        templates = self.templates.get("templates", {})
        
        if fallback_name in templates:
            return templates[fallback_name]
        
        # If fallback doesn't exist, return the first available template
        if templates:
            first_template = next(iter(templates.values()))
            logger.warning(f"Fallback template '{fallback_name}' not found, using first available template")
            return first_template
        
        # Last resort - return hardcoded minimal config
        logger.error("No templates available, using hardcoded fallback")
        return {
            "format": {
                "system_start": "<|im_start|>system\n",
                "system_end": "<|im_end|>\n",
                "user_start": "<|im_start|>user\n",
                "user_end": "<|im_end|>\n",
                "assistant_start": "<|im_start|>assistant\n",
                "assistant_end": "<|im_end|>\n"
            },
            "stop_tokens": ["<|im_end|>", "<|im_start|>"],
            "default_system_message": "You are a helpful AI assistant."
        }
    
    def get_stop_tokens_for_model(self, model_name: str) -> List[str]:
        """Get stop tokens for a specific model"""
        template_config = self.get_template_for_model(model_name)
        return template_config.get("stop_tokens", ["<|im_end|>"])
    
    def get_default_system_message(self, model_name: str) -> str:
        """Get default system message for a specific model"""
        template_config = self.get_template_for_model(model_name)
        return template_config.get(
            "default_system_message",
            self.templates.get("settings", {}).get("default_system_message", "You are a helpful AI assistant.")
        )
    
    def reload_templates(self):
        """Force reload of template configuration"""
        self._templates_cache = None
        logger.info("Template configuration cache cleared, will reload on next access")
    
    def list_available_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.get("templates", {}).keys())
    
    def validate_template_config(self) -> bool:
        """Validate that the template configuration is valid"""
        try:
            templates = self.templates.get("templates", {})
            
            if not templates:
                logger.error("No templates defined in configuration")
                return False
            
            for name, config in templates.items():
                # Check required fields
                if "format" not in config:
                    logger.error(f"Template '{name}' missing 'format' section")
                    return False
                
                format_config = config["format"]
                required_format_keys = ["system_start", "system_end", "user_start", "user_end", "assistant_start", "assistant_end"]
                
                for key in required_format_keys:
                    if key not in format_config:
                        logger.error(f"Template '{name}' format missing required key: '{key}'")
                        return False
                
                if "stop_tokens" not in config or not isinstance(config["stop_tokens"], list):
                    logger.error(f"Template '{name}' missing or invalid 'stop_tokens'")
                    return False
            
            logger.info("Template configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Template configuration validation failed: {e}")
            return False


# Global instance for convenient access
@lru_cache(maxsize=1)
def get_template_manager() -> TemplateManager:
    """Get a cached instance of the template manager"""
    return TemplateManager()


def load_templates() -> Dict[str, Any]:
    """Load chat templates from configuration (legacy function)"""
    manager = get_template_manager()
    return manager.templates 