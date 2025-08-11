import sys
import logging
from putergenai import PuterClient

def main():
    client = PuterClient()

    # Configure logging
    logger = logging.getLogger('putergenai.client')
    logger.handlers = []  # Clear existing handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(console_handler)

    # Login
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    try:
        client.login(username, password)
        print("Login successful!")
    except ValueError as e:
        print(f"Login failed: {e}")
        sys.exit(1)

    # Debugging option
    debug_input = input("Enable debug logging? (y/n): ").lower() == 'y'
    if not debug_input:
        logger.setLevel(logging.CRITICAL + 1)  # Suppress all logs
    else:
        logger.setLevel(logging.DEBUG)

    # Available models from model_to_driver
    available_models = list(client.model_to_driver.keys())
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    # Select model
    while True:
        try:
            model_choice = int(input("\nSelect a model (enter number): ")) - 1
            if 0 <= model_choice < len(available_models):
                break
            print(f"Please select a number between 1 and {len(available_models)}.")
        except ValueError:
            print("Please enter a valid number.")
    selected_model = available_models[model_choice]

    # Stream option
    stream_input = input("Enable streaming? (y/n): ").lower() == 'y'

    # Test mode option
    test_mode_input = input("Enable test mode? (y/n): ").lower() == 'y'

    # Strict model option
    strict_model_input = input("Enforce strict model usage? (y/n): ").lower() == 'y'

    # Temperature
    while True:
        try:
            temperature_input = float(input("Enter temperature (0-2, default 0.7): ") or 0.7)
            if 0 <= temperature_input <= 2:
                break
            print("Temperature must be between 0 and 2.")
        except ValueError:
            print("Please enter a valid number.")

    # Show used model option
    show_model_input = input("Show used model after response? (y/n): ").lower() == 'y'

    options = {
        "model": selected_model,
        "stream": stream_input,
        "temperature": temperature_input
    }

    # Chat history
    messages = []

    print("\nChat started. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        try:
            if stream_input:
                # Streaming response
                gen = client.ai_chat(messages=messages, options=options, test_mode=test_mode_input, strict_model=strict_model_input)
                print("Assistant: ", end='', flush=True)
                response_content = ''
                used_model = selected_model
                for content, model in gen:
                    print(content, end='', flush=True)
                    response_content += content
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
                print(f"Assistant: {content}")
                if show_model_input or debug_input:
                    print(f"Used model: {used_model}")
                messages.append({"role": "assistant", "content": content})
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()