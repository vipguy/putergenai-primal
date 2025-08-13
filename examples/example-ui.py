import sys
import logging
import re
import os
from putergenai import PuterClient
import requests
from PIL import Image, ImageDraw, ImageFont
import customtkinter as ctk
import tkinter as tk
import tkinter.messagebox as mbox
from cryptography.fernet import Fernet
import stat
import threading
import concurrent.futures

def generate_local_image(prompt, filename='local_image.png', bg_color='#496d89', font_size=32, text_color='#ffff00'):
    width, height = 600, 300
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, FileNotFoundError):
        font = ImageFont.load_default()
    # Center the text
    try:
        bbox = d.textbbox((0, 0), prompt, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow
        text_width, text_height = font.getsize(prompt)
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    d.text((x, y), prompt, fill=text_color, font=font)
    img.save(filename)
    return filename


def sanitize_string(s, allow_empty=False, allow_path=False):
    """
    Sanitize a string input.
    Parameters:
        s (str): Input string to sanitize.
        allow_empty (bool): If False, raises error on empty string.
        allow_path (bool): If True, allows path-like characters.
    Returns:
        str: Sanitized string.
    Raises:
        ValueError: If input is not valid.
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


# --- CustomTkinter GUI ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PuterApp(ctk.CTk):
    def _load_encryption_key(self):
        self._key_file = "api_keys.key"
        env_key = os.environ.get("PUTERGENAI_FERNET_KEY")
        if env_key:
            self._fernet_key = env_key.encode()
        elif os.path.exists(self._key_file):
            with open(self._key_file, "rb") as f:
                self._fernet_key = f.read()
        else:
            self._fernet_key = Fernet.generate_key()
            with open(self._key_file, "wb") as f:
                f.write(self._fernet_key)
            # Restrict permissions to user only (Windows)
            try:
                os.chmod(self._key_file, stat.S_IREAD | stat.S_IWRITE)
            except Exception as e:
                print(f"Warning: Could not set permissions on {self._key_file}: {e}")
        self._fernet = Fernet(self._fernet_key)

    def _encrypt(self, plaintext):
        return self._fernet.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext):
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def _load_api_keys(self):
        api_keys = {}
        if os.path.exists("api_keys.cfg"):
            try:
                with open("api_keys.cfg", "r") as f:
                    for line in f:
                        if "=" in line:
                            k, v = line.strip().split("=", 1)
                            try:
                                api_keys[k] = self._decrypt(v)
                            except Exception:
                                continue
            except Exception as e:
                print(f"Error loading API keys: {e}")
        return api_keys

    def _save_api_keys(self):
        """Save API keys to encrypted file"""
        try:
            # Create a secure temporary list to avoid direct string formatting
            encrypted_lines = []
            for key, value in self.api_keys.items():
                if value and value.strip():  # Only save non-empty keys
                    try:
                        encrypted_value = self._encrypt(value)
                        # Use safer string concatenation instead of f-string for sensitive data
                        line = key + "=" + encrypted_value + "\n"
                        encrypted_lines.append(line)
                    except Exception as encrypt_error:
                        print(f"Error encrypting key for service: {encrypt_error}")
                        continue
            
            # Write all encrypted lines at once
            config_file = "api_keys.cfg"
            with open(config_file, "w") as f:
                f.writelines(encrypted_lines)
            
            # Set restrictive file permissions (Windows compatible)
            try:
                os.chmod(config_file, stat.S_IREAD | stat.S_IWRITE)
            except Exception as perm_error:
                print(f"Warning: Could not set file permissions: {perm_error}")
                
        except Exception as e:
            print(f"Error saving API keys: {e}")

    def _build_main(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Model selection
        ctk.CTkLabel(self.main_frame, text="Select Model", font=("Arial", 18)).pack(pady=10)
        self.models = ["None (Disabled)"] + list(self.client.model_to_driver.keys())
        self.model_var = ctk.StringVar(value=self.models[1])  # Default to first real model, not "None"
        self.model_menu = ctk.CTkOptionMenu(self.main_frame, variable=self.model_var, values=[
            m if m == "None (Disabled)" else f"{m} {'[Image]' if m in self.image_models or any(x in m.lower() for x in ['vision','image','img']) else ''}" for m in self.models
        ])
        self.model_menu.pack(pady=5)

        # API selection dropdown
        ctk.CTkLabel(self.main_frame, text="Select Image Generation API", font=("Arial", 14)).pack(pady=5)
        self.api_var = ctk.StringVar(value=self.api_options[0])
        def on_api_select(choice):
            if choice == "None (Disabled)":
                self.api_key_section.pack_forget()
                self.status_label.configure(text="Image API disabled. Use local generation only.", text_color="orange")
                # Update button appearance to show it's disabled
                self.image_api_btn.configure(fg_color="#666666", text="Generate Image (API - Disabled)")
            elif choice in ["Hugging Face", "Replicate", "DeepAI", "OpenAI"]:
                mbox.showinfo("API Key Required", f"You selected {choice}. You will need to add an API key to use image generation.")
                try:
                    self.api_key_entry.delete(0, tk.END)
                    # Get the API key and ensure it's a string
                    api_key = self.api_keys.get(choice, "")
                    if api_key is None:
                        api_key = ""
                    self.api_key_entry.insert(0, str(api_key))
                except Exception as e:
                    print("Error setting API key entry field")
                    # If there's an error, just clear the field
                    try:
                        self.api_key_entry.delete(0, tk.END)
                    except:
                        pass
                self.api_key_section.pack(pady=5)
                self.status_label.configure(text=f"{choice} API selected.", text_color="blue")
                # Restore button appearance
                self.image_api_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"], text="Generate Image (API)")
            else:
                self.api_key_section.pack_forget()
                # Restore button appearance for other options
                self.image_api_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"], text="Generate Image (API)")
        self.api_menu = ctk.CTkOptionMenu(self.main_frame, variable=self.api_var, values=self.api_options, command=on_api_select)
        self.api_menu.pack(pady=5)

        # Section for API key entry
        self.api_key_section = ctk.CTkFrame(self.main_frame)
        ctk.CTkLabel(self.api_key_section, text="Enter API Key:").pack(side="left", padx=5)
        self.api_key_entry = ctk.CTkEntry(self.api_key_section, show="*")
        self.api_key_entry.pack(side="left", padx=5)
        def save_api_key():
            selected_api = self.api_var.get()
            key = self.api_key_entry.get()
            self.api_keys[selected_api] = key
            self._save_api_keys()
            mbox.showinfo("API Key Saved", f"API key for {selected_api} saved.")
        ctk.CTkButton(self.api_key_section, text="Save Key", command=save_api_key).pack(side="left", padx=5)

        # Chat and image controls
        self.chat_frame = ctk.CTkFrame(self.main_frame)
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat_box = ctk.CTkTextbox(self.chat_frame, width=700, height=300)
        self.chat_box.pack(pady=5)

        self.prompt_entry = ctk.CTkEntry(self.chat_frame, placeholder_text="Type your message or image prompt...")
        self.prompt_entry.pack(fill="x", pady=5)
        self.prompt_entry.bind("<Return>", lambda event: self._send_chat())

        self.button_frame = ctk.CTkFrame(self.chat_frame)
        self.button_frame.pack(fill="x", pady=5)
        ctk.CTkButton(self.button_frame, text="Send Chat", command=self._send_chat).pack(side="left", padx=5)
        self.image_api_btn = ctk.CTkButton(self.button_frame, text="Generate Image (API)", command=self._send_image_api)
        self.image_api_btn.pack(side="left", padx=5)
        ctk.CTkButton(self.button_frame, text="Generate Image (Local)", command=self._send_image_local).pack(side="left", padx=5)
        ctk.CTkButton(self.button_frame, text="Sign Out", command=self._sign_out, fg_color="#d9534f", text_color="white").pack(side="right", padx=5)
        
        # Status label for showing messages
        self.status_label = ctk.CTkLabel(self.chat_frame, text="", text_color="yellow")
        self.status_label.pack(pady=5)
    # Sign Out button will be added in _build_main

    def _sign_out(self):
        # Clear user session and return to login screen
        self.client = None
        self.selected_model = None
        self.messages = []
        self.options = {}
        self.api_online = None
        self.main_frame.pack_forget()
        self._build_login()
    def _get_api_key(self, api_name):
        # Check if we already have a valid API key
        existing_key = self.api_keys.get(api_name)
        if existing_key and existing_key.strip():  # Make sure it's not empty or whitespace
            return existing_key
            
        # Popup for API key entry (use customtkinter for consistency)
        key_win = ctk.CTkToplevel(self)
        key_win.title(f"Enter {api_name} API Key")
        key_win.geometry("350x120")
        ctk.CTkLabel(key_win, text=f"{api_name} API Key:").pack(pady=10)
        key_entry = ctk.CTkEntry(key_win, show="*")
        key_entry.pack(pady=5)
        result = {"key": None}
        
        def save_key():
            entered_key = key_entry.get()
            if entered_key and entered_key.strip():  # Only save if not empty
                result["key"] = entered_key.strip()
                self.api_keys[api_name] = result["key"]
                self._save_api_keys()
            key_win.destroy()
            
        ctk.CTkButton(key_win, text="Save", command=save_key).pack(pady=10)
        key_win.wait_window()
        return result["key"]

    def _send_image_api(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            self.status_label.configure(text="Prompt cannot be empty.")
            return
        api_choice = self.api_var.get()
        
        # Check if API is disabled
        if api_choice == "None (Disabled)":
            self.status_label.configure(text="Image API is disabled. Use local generation instead.", text_color="orange")
            mbox.showwarning("API Disabled", "Image API generation is disabled. Please use the 'Generate Image (Local)' button instead or select a different API.")
            return
        
        # Disable button and show progress
        self.image_api_btn.configure(state="disabled", text="Generating...")
        self.status_label.configure(text="Generating image, please wait...", text_color="blue")
        
        # Run image generation in background thread
        def generate_image_thread():
            try:
                self.notify_api_signup()
                
                if api_choice == "Hugging Face":
                    self._generate_huggingface_image(prompt)
                elif api_choice == "Replicate":
                    self._generate_replicate_image(prompt)
                elif api_choice == "DeepAI":
                    self._generate_deepai_image(prompt)
                elif api_choice == "OpenAI":
                    self._generate_openai_image(prompt)
                    
            except Exception as e:
                self.after(0, lambda: self._image_generation_error(f"Image API error: {e}"))
            finally:
                # Re-enable button in main thread
                self.after(0, lambda: self.image_api_btn.configure(state="normal", text="Generate Image (API)"))
        
        # Start generation in background
        threading.Thread(target=generate_image_thread, daemon=True).start()

    def _generate_huggingface_image(self, prompt):
        """Generate image using Hugging Face API"""
        key = self._get_api_key("Hugging Face")
        if not key:
            self.after(0, lambda: self.status_label.configure(text="No Hugging Face API key provided.", text_color="red"))
            return
            
        # Hugging Face with multiple model fallbacks
        models_to_try = [
            "black-forest-labs/FLUX.1-schnell",  # New fast model
            "stabilityai/stable-diffusion-xl-base-1.0",
            "CompVis/stable-diffusion-v1-4", 
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1"
        ]
        
        success = False
        last_error = None
        
        for i, model_name in enumerate(models_to_try):
            self.after(0, lambda m=model_name: self.status_label.configure(text=f"Trying model {i+1}/{len(models_to_try)}: {m}", text_color="blue"))
            
            url = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {"Authorization": f"Bearer {key}"}
            data = {"inputs": prompt}
            
            try:
                import requests
                resp = requests.post(url, headers=headers, json=data, timeout=60)  # Increased timeout
                
                # Debug: Log response details (without sensitive data)
                print(f"Model {model_name}: Status {resp.status_code}")
                if resp.status_code != 200:
                    # Don't log the actual response as it may contain sensitive error details
                    print(f"Request failed with status {resp.status_code}")
                
                if resp.status_code == 200:
                    # Check if response is actually an image
                    content_type = resp.headers.get('content-type', '')
                    if content_type.startswith('image/'):
                        # Save image from response
                        fname = f"hf_image_{model_name.split('/')[-1]}.png"
                        with open(fname, "wb") as f:
                            f.write(resp.content)
                        self.after(0, lambda m=model_name, f=fname: self.chat_box.insert("end", f"[Hugging Face] Prompt: {prompt}\nModel: {m}\nSaved as: {f}\n\n"))
                        self.after(0, lambda f=fname: self.status_label.configure(text=f"Image saved as {f}.", text_color="green"))
                        success = True
                        break
                    else:
                        # Response might be JSON with error or loading message
                        try:
                            error_data = resp.json()
                            if 'estimated_time' in error_data:
                                wait_time = error_data['estimated_time']
                                self.after(0, lambda m=model_name, t=wait_time: self.chat_box.insert("end", f"[Hugging Face] Model {m} is loading (estimated {t}s), trying next model...\n"))
                                last_error = f"Model loading: {wait_time}s"
                                continue
                            elif 'error' in error_data:
                                last_error = "API returned an error"
                                print("API Error occurred (details not logged for security)")
                                continue
                        except:
                            last_error = "Invalid response format from API"
                            continue
                elif resp.status_code == 401:
                    last_error = "Invalid API token - check your Hugging Face token"
                    break  # No point trying other models with bad token
                elif resp.status_code == 404:
                    last_error = f"Model {model_name} not found"
                    continue
                elif resp.status_code == 429:
                    last_error = "Rate limit exceeded - wait a few minutes"
                    continue
                elif resp.status_code == 503:
                    last_error = f"Model {model_name} unavailable (service overloaded)"
                    continue
                else:
                    last_error = f"HTTP {resp.status_code} error from API"
                    continue
                    
            except requests.exceptions.Timeout:
                last_error = f"Timeout waiting for {model_name}"
                continue
            except requests.exceptions.ConnectionError:
                last_error = "Network connection error"
                continue
            except Exception as e:
                last_error = f"Error with {model_name}: {str(e)}"
                continue
        
        if not success:
            # Show specific error message
            error_msg = f"All Hugging Face models failed. Last error: {last_error}"
            self.after(0, lambda: self.chat_box.insert("end", f"[Hugging Face] {error_msg}\n\nTroubleshooting:\n"))
            self.after(0, lambda: self.chat_box.insert("end", f"1. Verify your API token at https://huggingface.co/settings/tokens\n"))
            self.after(0, lambda: self.chat_box.insert("end", f"2. Check if your token has 'Read' permission\n"))
            self.after(0, lambda: self.chat_box.insert("end", f"3. Try again in a few minutes (models may be loading)\n"))
            self.after(0, lambda: self.chat_box.insert("end", f"4. Use 'Local' generation as an alternative\n\n"))
            self.after(0, lambda: self.status_label.configure(text="All Hugging Face models failed. Check console for details.", text_color="red"))

    def _generate_replicate_image(self, prompt):
        """Generate image using Replicate API"""
        key = self._get_api_key("Replicate")
        if not key:
            self.after(0, lambda: self.status_label.configure(text="No Replicate API key provided.", text_color="red"))
            return
        
        url = "https://api.replicate.com/v1/predictions"
        headers = {"Authorization": f"Token {key}", "Content-Type": "application/json"}
        data = {
            "version": "a9758cb8c6c7e0e6c7e0e6c7e0e6c7e0e6c7e0e6c7e0e6c7e0e6c7e0e6c7e0e6", # SDXL
            "input": {"prompt": prompt}
        }
        import requests
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code == 201:
            output = resp.json()["urls"]["get"]
            self.after(0, lambda: self.chat_box.insert("end", f"[Replicate] Prompt: {prompt}\nResult URL: {output}\n\n"))
            self.after(0, lambda: self.status_label.configure(text="Replicate request submitted.", text_color="green"))
        else:
            self.after(0, lambda: self.chat_box.insert("end", f"[Replicate] Error: API request failed\n\n"))
            self.after(0, lambda: self.status_label.configure(text="Replicate API error.", text_color="red"))

    def _generate_deepai_image(self, prompt):
        """Generate image using DeepAI API"""
        key = self._get_api_key("DeepAI")
        if not key:
            self.after(0, lambda: self.status_label.configure(text="No DeepAI API key provided.", text_color="red"))
            return
        
        url = "https://api.deepai.org/api/text2img"
        headers = {"api-key": key}
        data = {"text": prompt}
        resp = requests.post(url, headers=headers, data=data)
        if resp.status_code == 200:
            output = resp.json().get("output_url")
            self.after(0, lambda: self.chat_box.insert("end", f"[DeepAI] Prompt: {prompt}\nImage URL: {output}\n\n"))
            self.after(0, lambda: self.status_label.configure(text="DeepAI image generated.", text_color="green"))
        else:
            self.after(0, lambda: self.chat_box.insert("end", f"[DeepAI] Error: API request failed\n\n"))
            self.after(0, lambda: self.status_label.configure(text="DeepAI API error.", text_color="red"))

    def _generate_openai_image(self, prompt):
        """Generate image using OpenAI API"""
        key = self._get_api_key("OpenAI")
        if not key:
            self.after(0, lambda: self.status_label.configure(text="No OpenAI API key provided.", text_color="red"))
            return
        
        url = "https://api.openai.com/v1/images/generations"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {"prompt": prompt, "n": 1, "size": "512x512"}
        import requests
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code == 200:
            output = resp.json()["data"][0]["url"]
            self.after(0, lambda: self.chat_box.insert("end", f"[OpenAI] Prompt: {prompt}\nImage URL: {output}\n\n"))
            self.after(0, lambda: self.status_label.configure(text="OpenAI image generated.", text_color="green"))
        else:
            self.after(0, lambda: self.chat_box.insert("end", f"[OpenAI] Error: API request failed\n\n"))
            self.after(0, lambda: self.status_label.configure(text="OpenAI API error.", text_color="red"))

    def _image_generation_error(self, error_msg):
        """Handle image generation errors in main thread"""
        self.status_label.configure(text=error_msg, text_color="red")
        self.chat_box.insert("end", f"[Image API] Error: {error_msg}\n\n")
    def notify_api_signup(self):
        mbox.showinfo(
            "Free API Notice",
            "To use most free image generation APIs (Hugging Face, Replicate, OpenAI, etc.), you will likely need to create a free account and obtain an API key. DeepAI may work without signup for basic use."
        )
    def __init__(self):
        super().__init__()
        
        # Initialize encryption first
        self._load_encryption_key()
        
        self.title("PuterGenAI Chat & Image GUI")
        self.geometry("1000x800")
        self.resizable(False, False)
        self.client = None
        self.models = []
        self.image_models = [
                'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
                'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-chat-latest',
                'claude-sonnet-4', 'claude-opus-4', 'claude-3-7-sonnet', 'claude-3-5-sonnet',
                'gemini-2.0-flash', 'gemini-1.5-flash', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                'mistral-large-latest', 'pixtral-large-latest', 'codestral-latest', 'google/gemma-2-27b-it', 'grok-beta'
            ]
        self.selected_model = None
        self.messages = []
        self.options = {}
        self.api_online = None
        
        # Load API keys using encryption
        self.api_keys = self._load_api_keys()
        self.api_options = ["None (Disabled)", "Hugging Face", "Replicate", "DeepAI", "OpenAI"]
        self._build_menu_native()
        self._build_login()

    def _build_menu_native(self):
        self.menu_native = tk.Menu(self)
        # File menu
        file_menu = tk.Menu(self.menu_native, tearoff=0)
        file_menu.add_command(label="Exit", command=self._exit_app)
        self.menu_native.add_cascade(label="File", menu=file_menu)
        # Info menu
        info_menu = tk.Menu(self.menu_native, tearoff=0)
        info_menu.add_command(label="About", command=self._show_info)
        self.menu_native.add_cascade(label="Info", menu=info_menu)
        # API Status menu
        api_menu = tk.Menu(self.menu_native, tearoff=0)
        api_menu.add_command(label="Check Image API", command=self._check_api_status)
        self.menu_native.add_cascade(label="API Status", menu=api_menu)
        self.configure(menu=self.menu_native)

    def _exit_app(self):
        self.destroy()

    def _check_api_status(self):
        # Try a dummy image prompt to check API status
        try:
            image_url = self.client.ai_txt2img("test") if self.client else None
            if image_url:
                mbox.showinfo("API Status", "Image API is ONLINE.")
            else:
                mbox.showwarning("API Status", "Image API is OFFLINE or unavailable.")
        except Exception as e:
            mbox.showwarning("API Status", f"Image API is OFFLINE or unavailable.\nError: {e}")

    # Removed simulated menubar frame

    def _show_info(self):
        info = (
            "PuterGenAI GUI\n"
            "- Login with your Puter credentials.\n"
            "- Select a model (models marked [Image] may support image generation).\n"
            "- Chat, generate images via API or locally.\n"
            "- API image generation may be unavailable for some models/accounts.\n"
            "- Local image generation always works.\n"
            "- Status and errors are shown below the chat box."
        )
        mbox.showinfo("Info", info)

    def _build_login(self):
        self.login_frame = ctk.CTkFrame(self)
        self.login_frame.pack(expand=True)
        ctk.CTkLabel(self.login_frame, text="Login to PuterGenAI", font=("Arial", 24)).pack(pady=20)
        self.username_entry = ctk.CTkEntry(self.login_frame, placeholder_text="Username")
        self.username_entry.pack(pady=10)
        self.password_entry = ctk.CTkEntry(self.login_frame, placeholder_text="Password", show="*")
        self.password_entry.pack(pady=10)
        
        # Add Enter key binding for password field
        self.password_entry.bind("<Return>", lambda event: self._login())
        
        self.login_status = ctk.CTkLabel(self.login_frame, text="", text_color="red")
        self.login_status.pack(pady=5)
        
        self.login_button = ctk.CTkButton(self.login_frame, text="Login", command=self._login)
        self.login_button.pack(pady=20)
        
        # Cancel button (initially hidden)
        self.cancel_button = ctk.CTkButton(self.login_frame, text="Cancel", command=self._cancel_login, fg_color="#d9534f")
        
        # Progress bar for login
        self.login_progress = ctk.CTkProgressBar(self.login_frame)
        self.login_progress.set(0)
        
        # Login thread reference for cancellation
        self.login_thread = None

    def _login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        
        # Validate inputs
        if not username or not password:
            self.login_status.configure(text="Please enter both username and password.", text_color="red")
            return
        
        # Disable login button and show progress
        self.login_button.configure(state="disabled", text="Logging in...")
        self.cancel_button.pack(pady=5)
        self.login_status.configure(text="Connecting...", text_color="blue")
        self.login_progress.pack(pady=10)
        self.login_progress.start()
        
        # Run login in separate thread to prevent UI freezing
        def login_thread():
            try:
                # Sanitize inputs
                safe_username = sanitize_string(username)
                safe_password = sanitize_string(password)
                
                # Create client and login with timeout handling
                client = PuterClient()
                
                # Update status
                self.after(0, lambda: self.login_status.configure(text="Authenticating...", text_color="blue"))
                
                client.login(safe_username, safe_password)
                
                # Schedule UI update in main thread
                self.after(0, lambda: self._login_success(client))
                
            except Exception as e:
                # Schedule error update in main thread
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    error_msg = "Connection timeout. Please check your internet connection and try again."
                elif "connection" in error_msg.lower():
                    error_msg = "Unable to connect to server. Please check your internet connection."
                elif "authentication" in error_msg.lower() or "login" in error_msg.lower():
                    error_msg = "Invalid username or password."
                
                self.after(0, lambda: self._login_error(error_msg))
        
        # Start login thread
        self.login_thread = threading.Thread(target=login_thread, daemon=True)
        self.login_thread.start()

    def _cancel_login(self):
        """Cancel the login process"""
        self.login_progress.stop()
        self.login_progress.pack_forget()
        self.cancel_button.pack_forget()
        self.login_button.configure(state="normal", text="Login")
        self.login_status.configure(text="Login cancelled.", text_color="orange")
        # Note: We can't actually stop the thread, but we can ignore its results

    def _login_success(self, client):
        """Handle successful login in main thread"""
        self.client = client
        self.login_progress.stop()
        self.login_progress.pack_forget()
        self.cancel_button.pack_forget()
        self.login_status.configure(text="Login successful!", text_color="green")
        
        # Skip image access check for now to avoid hanging
        # User can test image generation manually after login
        note = "Login successful! You can test image generation using the API buttons after login."
        mbox.showinfo("Login Successful", note)
        self.login_frame.pack_forget()
        self._build_main()

    def _complete_login(self, has_image_access):
        """Complete login process with image access info"""
        note = "You have access to image generation via API." if has_image_access else "You do NOT have access to image generation via API. Use local image generation instead."
        mbox.showinfo("Image Generation Access", note)
        self.login_frame.pack_forget()
        self._build_main()

    def _login_error(self, error_msg):
        """Handle login error in main thread"""
        self.login_progress.stop()
        self.login_progress.pack_forget()
        self.cancel_button.pack_forget()
        self.login_button.configure(state="normal", text="Login")
        self.login_status.configure(text=f"Login failed: {error_msg}", text_color="red")


    def _send_chat(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            self.status_label.configure(text="Prompt cannot be empty.")
            return
            
        # Check if model is disabled
        selected_model_text = self.model_var.get()
        if selected_model_text == "None (Disabled)":
            self.status_label.configure(text="Chat model is disabled. Please select a model first.", text_color="orange")
            return
            
        self.selected_model = selected_model_text.split()[0]
        self.options = {
            "model": self.selected_model,
            "stream": False,
            "temperature": 0.7
        }
        self.messages.append({"role": "user", "content": prompt})
        try:
            result = self.client.ai_chat(messages=self.messages, options=self.options)
            content = result["response"].get('result', {}).get('message', {}).get('content', '')
            used_model = result["used_model"]
            self.chat_box.insert("end", f"You: {prompt}\nAssistant: {content}\n[Model: {used_model}]\n\n")
            self.messages.append({"role": "assistant", "content": content})
            self.status_label.configure(text="Chat response received.", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Chat error: {e}", text_color="red")

    def _send_image(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            self.status_label.configure(text="Prompt cannot be empty.")
            return
        try:
            image_url = self.client.ai_txt2img(prompt)
            if image_url:
                self.api_online = True
                self.chat_box.insert("end", f"[Image API] Prompt: {prompt}\nImage URL: {image_url}\n\n")
                self.status_label.configure(text="Image generated via API.", text_color="green")
                mbox.showinfo("API Status", "Image API is ONLINE.")
            else:
                self.api_online = False
                self.chat_box.insert("end", f"[Image API] Prompt: {prompt}\nImage generation failed.\n\n")
                self.status_label.configure(text="Image generation failed.", text_color="red")
                mbox.showwarning("API Status", "Image API is OFFLINE or unavailable.")
        except Exception as e:
            self.api_online = False
            self.chat_box.insert("end", f"[Image API] Error: {e}\n\n")
            self.status_label.configure(text=f"Image API error: {e}", text_color="red")
            mbox.showwarning("API Status", f"Image API is OFFLINE or unavailable.\nError: {e}")

    def _send_image_local(self):
        self.notify_api_signup()
        prompt = self.prompt_entry.get()
        if not prompt:
            self.status_label.configure(text="Prompt cannot be empty.")
            return
        # Popup for options
        options_win = tk.Toplevel(self)
        options_win.title("Local Image Options")
        options_win.geometry("400x350")

        bg_label = tk.Label(options_win, text="Background Color (hex)")
        bg_label.pack(pady=5)
        bg_entry = tk.Entry(options_win)
        bg_entry.insert(0, "#496d89")
        bg_entry.pack(pady=5)

        font_label = tk.Label(options_win, text="Font Size")
        font_label.pack(pady=5)
        font_entry = tk.Entry(options_win)
        font_entry.insert(0, "32")
        font_entry.pack(pady=5)

        text_label = tk.Label(options_win, text="Text Color (hex)")
        text_label.pack(pady=5)
        text_entry = tk.Entry(options_win)
        text_entry.insert(0, "#ffff00")
        text_entry.pack(pady=5)

        file_label = tk.Label(options_win, text="Filename")
        file_label.pack(pady=5)
        file_entry = tk.Entry(options_win)
        file_entry.insert(0, "local_image.png")
        file_entry.pack(pady=5)

        def generate_and_close():
            bg_color = bg_entry.get() or "#496d89"
            try:
                font_size = int(font_entry.get() or 32)
            except ValueError:
                font_size = 32
            text_color = text_entry.get() or "#ffff00"
            filename = file_entry.get() or "local_image.png"
            fname = generate_local_image(prompt, filename, bg_color, font_size, text_color)
            self.chat_box.insert("end", f"[Image Local] Prompt: {prompt}\nSaved as: {fname}\n\n")
            self.status_label.configure(text=f"Local image saved as {fname}.", text_color="green")
            options_win.destroy()

        gen_btn = tk.Button(options_win, text="Generate", command=generate_and_close)
        gen_btn.pack(pady=10)

# --- Flask Secure Cookie Example ---
from flask import Flask, make_response, request
from cryptography.fernet import Fernet
import secrets

app = Flask("Secure Example")
# Use a secure key from environment or generate securely
app.secret_key = secrets.token_hex(32)
fernet = Fernet(Fernet.generate_key())

@app.route('/')
def index():
    password = request.args.get("password")
    if password:
        # Don't store passwords in cookies at all - this is just for demonstration
        # In production, use session tokens or other secure methods
        session_token = secrets.token_urlsafe(32)
        
        # Store the password securely server-side (not shown here)
        # encrypted_password = fernet.encrypt(password.encode()).decode()
        
        resp = make_response("Authentication token created (password not stored in cookie)")
        # Store only a session token, not the actual password
        resp.set_cookie("session_token", session_token, 
                       secure=True, 
                       httponly=True, 
                       samesite='Strict',
                       max_age=3600)  # 1 hour expiration
        return resp
    return "No password provided"

# Uncomment below to run Flask app
# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    PuterApp().mainloop()
