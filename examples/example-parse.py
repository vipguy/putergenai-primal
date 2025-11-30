#!/usr/bin/env python3
import sys
import logging
import re
import asyncio
import time
import os
from getpass import getpass
from collections import Counter

# Add the src directory to the Python path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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

async def main():
    # Option to ignore SSL for debugging "Semaphore timeout" issues
    ignore_ssl_input = input("Ignore SSL verification (try 'y' if you get timeouts)? (y/n): ").strip().lower() == 'y'

    # Use async context manager for proper session handling
    async with PuterClient(ignore_ssl=ignore_ssl_input) as client:
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
                # Await the login method
                await client.login(username, password)
                print("Login successful!")
                break
            except ValueError as e:
                attempts += 1
                logging.warning(f"Failed login attempt {attempts} for user: {username}")
                print(f"Login failed: {e}")
                if attempts >= max_attempts:
                    print("Maximum login attempts exceeded. Exiting.")
                    sys.exit(1)

        # Retrieving current models from API
        print("\n Retrieving current models from API ")
        available_models = []  # Will be defined here
        has_dynamic_api = hasattr(client, 'get_available_models')
        if has_dynamic_api:
            try:
                models_data = await client.get_available_models()
                available_models = models_data.get('models', [])
                total_models = len(available_models)
                print(f"✓ Successfully retrieved {total_models} models")

                # Optionally: show first 10
                show_preview = input("Show first 10 models? (y/n): ").strip().lower() == 'y'
                if show_preview and available_models:
                    for i, model in enumerate(available_models[:10], 1):
                        print(f"  {i}. {model}")
            except Exception as e:
                print(f"✗ Error retrieving models: {e}")
                print("Static model list will be used")
                available_models = list(client.model_to_driver.keys())
        else:
            print("✗ Dynamic API unavailable, using static list")
            available_models = list(client.model_to_driver.keys())
            print(f"✓ Found {len(available_models)} static models")

        # Updating driver mappings
        print("\n Updating model mappings ")
        has_mappings_api = hasattr(client, 'update_model_mappings')
        if has_mappings_api:
            update_mappings = input("Update model→driver mappings from API? (y/n): ").strip().lower() == 'y'

            if update_mappings:
                try:
                    old_count = len(client.model_to_driver)
                    await client.update_model_mappings()
                    new_count = len(client.model_to_driver)
                    print(f"✓ Mappings before: {old_count}, after: {new_count}")
                    print(f"  Added/updated: {new_count - old_count}")
                except Exception as e:
                    print(f"✗ Update error: {e}")
        else:
            print("✗ Mapping update unavailable")

        # Model filtering
        print("\n Model filtering ")
        print("1. Show all models")
        print("2. Filter by provider (openrouter, togetherai, claude etc.)")
        print("3. Filter by driver (openai-completion, together-ai etc.)")
        print("4. Search by name")
        print("5. Show only free models (with :free)")

        filter_choice = input("Choose option (1-5, Enter to skip): ").strip()

        filtered_models = []
        if filter_choice == '1':
            filtered_models = available_models
        elif filter_choice == '2':
            provider = input("Enter provider prefix (e.g., openrouter): ").strip()
            filtered_models = [m for m in available_models if m.startswith(provider)]
        elif filter_choice == '3':
            driver = input("Enter driver (e.g., openai-completion): ").strip()
            filtered_models = [m for m, d in client.model_to_driver.items() if d == driver]
        elif filter_choice == '4':
            search_term = input("Enter part of model name: ").strip().lower()
            filtered_models = [m for m in available_models if search_term in m.lower()]
        elif filter_choice == '5':
            filtered_models = [m for m in available_models if ':free' in m]
        else:
            filtered_models = available_models

        print(f"\nFound {len(filtered_models)} models:")
        for i, model in enumerate(filtered_models[:50], 1):  # Show first 50
            driver = client.model_to_driver.get(model, "unknown")
            print(f"{i}. {model} [{driver}]")

        if len(filtered_models) > 50:
            print(f"... and {len(filtered_models) - 50} more models")

        # Model availability check
        print("\n Model availability check ")
        has_model_check = hasattr(client, 'is_model_available')
        if has_model_check:
            check_model = input("Enter model name to check (Enter to skip): ").strip()

            if check_model:
                try:
                    is_available = await client.is_model_available(check_model)
                    if is_available:
                        driver = client.model_to_driver.get(check_model, "not defined")
                        print(f"✓ Model '{check_model}' is available")
                        print(f"  Driver: {driver}")
                    else:
                        print(f"✗ Model '{check_model}' not found in available list")
                        # Suggest similar
                        similar = [m for m in available_models if check_model.lower() in m.lower()]
                        if similar[:5]:
                            print("  Perhaps you meant:")
                            for s in similar[:5]:
                                print(f"    - {s}")
                except Exception as e:
                    print(f"✗ Check error: {e}")
        else:
            print("Model availability check unavailable")

        # Statistics
        print("\n Model Statistics ")
        show_stats = input("Show statistics by drivers? (y/n): ").strip().lower() == 'y'

        if show_stats:
            driver_stats = Counter(client.model_to_driver.values())

            print(f"\nTotal models: {len(client.model_to_driver)}")
            print(f"Unique drivers: {len(driver_stats)}")
            print("\nDistribution by drivers:")
            for driver, count in driver_stats.most_common():
                print(f"  {driver}: {count} models")

            # Top providers
            provider_counts = {}
            for model in available_models:
                if ':' in model:
                    provider = model.split(':')[0]
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1

            if provider_counts:
                print("\nTop-5 providers:")
                for provider, count in sorted(provider_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {provider}: {count} models")

        # Caching demonstration
        print("\n Caching test ")
        if has_dynamic_api:
            test_cache = input("Test caching? (y/n): ").strip().lower() == 'y'

            if test_cache:
                print("First request (without cache)...")
                start = time.time()
                await client.get_available_models(force_refresh=True)
                first_time = time.time() - start
                print(f"  Time: {first_time:.3f} sec")

                print("Second request (from cache)...")
                start = time.time()
                await client.get_available_models()
                cached_time = time.time() - start
                print(f"  Time: {cached_time:.3f} sec")
                if cached_time > 0:
                    print(f"  Acceleration: {first_time/cached_time:.1f}x")
                else:
                    print("  Acceleration: instantaneous")

                print("Third request (forced refresh)...")
                start = time.time()
                await client.get_available_models(force_refresh=True)
                refresh_time = time.time() - start
                print(f"  Time: {refresh_time:.3f} sec")
        else:
            print("Caching test unavailable (no dynamic API)")

        # Debugging option
        debug_input = input("Enable debug logging? (y/n): ").strip().lower() == 'y'
        if not debug_input:
            logger.setLevel(logging.CRITICAL + 1)  # Suppress all logs
        else:
            logger.setLevel(logging.DEBUG)

        # Available models from filtered_models or all available
        if not filtered_models:
            filtered_models = available_models

        # Whitelist models for selection - show with drivers
        print("\nAvailable models:")
        for i, model in enumerate(filtered_models, 1):
            driver = client.model_to_driver.get(model, "unknown")
            print(f"{i}. {model} [{driver}]")

        # Select model from filtered list
        while True:
            try:
                model_input = input("\nSelect a model (enter number): ").strip()
                model_choice = int(model_input) - 1
                if 0 <= model_choice < len(filtered_models):
                    selected_model = filtered_models[model_choice]
                    break
                print(f"Please select a number between 1 and {len(filtered_models)}.")
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
                    # Streaming response - await the call, then async iterate
                    gen = await client.ai_chat(
                        messages=messages,
                        options=options,
                        test_mode=test_mode_input,
                        strict_model=strict_model_input
                    )

                    print("Assistant: ", end='', flush=True)
                    response_content = ''
                    used_model = selected_model

                    async for content, model in gen:
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
                    # Non-streaming - await the call
                    result = await client.ai_chat(
                        messages=messages,
                        options=options,
                        test_mode=test_mode_input,
                        strict_model=strict_model_input
                    )

                    # Extract content based on Puter API response structure
                    content = ""
                    if 'result' in result["response"]:
                        res_data = result["response"]["result"]
                        if 'message' in res_data:
                            message = res_data['message']
                            if isinstance(message, dict):
                                content = message.get('content', '')
                            else:
                                content = str(message)
                        elif 'choices' in res_data: # OpenAI style fallback
                             content = res_data['choices'][0]['message']['content']

                    used_model = result["used_model"]
                    safe_content = str(content).replace('\x1b', '')
                    print(f"Assistant: {safe_content}")
                    if show_model_input or debug_input:
                        print(f"Used model: {used_model}")
                    messages.append({"role": "assistant", "content": safe_content})
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(0)
