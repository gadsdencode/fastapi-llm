#!/usr/bin/env python3

from app.config import get_template_manager

def test_template_matching():
    manager = get_template_manager()
    model_name = 'TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF'
    
    print(f"Testing template matching for: {model_name}")
    template = manager.get_template_for_model(model_name)
    
    if template:
        print("✓ Template found!")
        format_config = template.get("format", {})
        print(f"Conversation start: {repr(format_config.get('conversation_start', ''))}")
        print(f"User start: {repr(format_config.get('user_start', ''))}")
        print(f"Assistant start: {repr(format_config.get('assistant_start', ''))}")
        print(f"Stop tokens: {template.get('stop_tokens', [])}")
        
        # Test a sample conversation formatting
        from app.schemas.models import ChatMessage
        messages = [
            ChatMessage(role="user", content="Hello, how are you?")
        ]
        
        print("\nSample formatted conversation:")
        formatted_parts = []
        if 'conversation_start' in format_config:
            formatted_parts.append(format_config['conversation_start'])
        
        for message in messages:
            role = message.role
            content = message.content
            
            if f"{role}_start" in format_config:
                formatted_parts.append(format_config[f"{role}_start"])
            
            formatted_parts.append(content)
            
            if f"{role}_end" in format_config:
                formatted_parts.append(format_config[f"{role}_end"])
        
        if 'assistant_start' in format_config:
            formatted_parts.append(format_config['assistant_start'])
        
        formatted_prompt = ''.join(formatted_parts)
        print(repr(formatted_prompt))
        
    else:
        print("✗ No template found!")

if __name__ == "__main__":
    test_template_matching() 