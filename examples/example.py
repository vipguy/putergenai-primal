import sys
import logging
import re
from getpass import getpass
from putergenai import PuterClient

def sanitize_string(s, allow_empty=False, allow_path=False):
    """
    Sanitize user input for usernames, passwords, and file paths only.
    For chat and prompt text, allow natural language (no sanitization).
    """
    if not isinstance(s, str):
        raise ValueError("Input must be a string.")
    s = s.strip()
    if not allow_empty and not s:
        raise ValueError("Input cannot be empty.")
    if allow_path:
        pattern = r'^[\w\-\.\/]+$'
        if not re.match(pattern, s):
            raise ValueError("Input contains invalid characters for path.")
    return s

def sanitize_float(val, min_value=0.0, max_value=2.0, default=0.7):
    try:
        f = float(val)
    except (ValueError, TypeError):
        f = default
    if f < min_value or f > max_value:
        raise ValueError(f"Value must be between {min_value} and {max_value}.")
    return f

def main():
    client = PuterClient()

    # Configure logging to a file to avoid interleaving with console output
    logger = logging.getLogger('putergenai.client')
    logger.handlers = []  # Clear existing handlers
    # Log to a file instead of stdout
    file_handler = logging.FileHandler('putergenai.log')
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(file_handler)

    # Login
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        username = input("Enter your username: ")
        # Use getpass to mask password input
        password = getpass("Enter your password: ")
        try:
            username = sanitize_string(username)
            password = sanitize_string(password)
            client.login(username, password)
            print("Login successful!")
            break
        except ValueError as e:
            attempts += 1
            logging.warning(f"Failed login attempt {attempts} for user: {username}")
            print(f"Login failed: {e}")
            if attempts >= max_attempts:
                print("Maximum login attempts exceeded. Exiting.")
                sys.exit(1)

    # Debugging option
    debug_input = input("Enable debug logging? (y/n): ").strip().lower() == 'y'
    if not debug_input:
        logger.setLevel(logging.CRITICAL + 1)  # Suppress all logs
    else:
        logger.setLevel(logging.DEBUG)

    # Available models from model_to_driver
    available_models = list(client.model_to_driver.keys())
    # Whitelist models for selection
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    # Select model
    while True:
        try:
            model_input = input("\nSelect a model (enter number): ").strip()
            model_choice = int(model_input) - 1
            if 0 <= model_choice < len(available_models):
                selected_model = available_models[model_choice]
                break
            print(f"Please select a number between 1 and {len(available_models)}.")
        except ValueError:
            print("Please enter a valid number.")

    # Stream option
    stream_input = input("Enable streaming? (y/n): ").strip().lower() == 'y'

    # Test mode option
    test_mode_input = input("Enable test mode? (y/n): ").strip().lower() == 'y'

    # Strict model option
    strict_model_input = input("Enforce strict model usage? (y/n): ").strip().lower() == 'y'

    # Temperature
    while True:
        try:
            temp_raw = input("Enter temperature (0-2, default 0.7): ").strip()
            temperature_input = sanitize_float(temp_raw or 0.7)
            break
        except ValueError as e:
            print(f"Temperature error: {e}")

    # Show used model option
    show_model_input = input("Show used model after response? (y/n): ").strip().lower() == 'y'

    options = {
        "model": selected_model,
        "stream": stream_input,
        "temperature": temperature_input
    }

    # Chat history
    messages = []

    print("\nChat started. Type 'exit' to quit.")
    # Do not log sensitive info
    logging.getLogger('putergenai.client').propagate = False

    while True:
        user_input = input("\nYou: ")
        if user_input.lower().strip() == 'exit':
            break

        # For chat, allow natural language (no sanitization)
        messages.append({"role": "user", "content": user_input})

        try:
            if stream_input:
                # Streaming response
                gen = client.ai_chat(messages=messages, options=options, test_mode=test_mode_input, strict_model=strict_model_input)
                print("Assistant: ", end='', flush=True)
                response_content = ''
                used_model = selected_model
                for content, model in gen:
                    if content:  # Skip empty content
                        safe_content = str(content).replace('\x1b', '')
                        print(safe_content, end='', flush=True)
                        response_content += safe_content
                        used_model = model
                print()  # New line after stream
                if show_model_input or debug_input:
                    print(f"Used model: {used_model}")
                messages.append({"role": "assistant", "content": response_content})
            else:
                # Non-streaming
                result = client.ai_chat(messages=messages, options=options, test_mode=test_mode_input, strict_model=strict_model_input)
                content = result["response"].get('result', {}).get('message', {}).get('content', '')
                used_model = result["used_model"]
                safe_content = str(content).replace('\x1b', '')
                print(f"Assistant: {safe_content}")
                if show_model_input or debug_input:
                    print(f"Used model: {used_model}")
                messages.append({"role": "assistant", "content": safe_content})
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
