from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import openai  # Replace 'from openai import OpenAI' with this import
import os
import json
import logging
import zipfile
from io import BytesIO
import requests  # Add this import for making HTTP requests
import tempfile
import shutil
import subprocess

app = Flask(__name__, template_folder='.', static_folder='.')  # Set template folder to current directory
CORS(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Store files in memory for simplicity
files = {}

VERCEL_TOKEN = os.getenv("VERCEL_TOKEN")  # Ensure the Vercel token is loaded

# Chatbot memory file
CHATBOT_MEMORY_FILE = "chatbot_memory.json"

# Load chatbot memory from a local file
def load_chatbot_memory():
    if os.path.exists(CHATBOT_MEMORY_FILE):
        with open(CHATBOT_MEMORY_FILE, "r") as file:
            return json.load(file)
    return []

# Save chatbot memory to a local file
def save_chatbot_memory(memory):
    with open(CHATBOT_MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=2)

# Initialize chatbot memory
chatbot_memory = load_chatbot_memory()

@app.route('/favicon.ico')
def favicon():
    # Serve logo.png as favicon.ico for browser tab icon
    return send_from_directory(
        os.path.join(app.root_path, ''),
        'logo.png',
        mimetype='image/png'
    )

@app.route('/favicon.png')
def favicon_png():
    # Also serve logo.png as favicon.png for browsers that request it
    return send_from_directory(
        os.path.join(app.root_path, ''),
        'logo.png',
        mimetype='image/png'
    )

@app.route('/logo.png')
def logo():
    return send_from_directory(
        os.path.join(app.root_path, ''),
        'logo.png',
        mimetype='image/png'
    )

@app.route('/')
def home():
    return render_template('main.html')

def sanitize_response(response_text):
    """
    Sanitize the response text to ensure it is valid JSON.
    - If the response is already valid JSON, return it as is.
    - Otherwise, attempt to fix common issues like improperly escaped characters.
    """
    try:
        # Check if the response is already valid JSON
        json.loads(response_text)  # This will raise an error if the JSON is invalid
        return response_text
    except json.JSONDecodeError:
        logging.warning("Response is not valid JSON. Attempting to sanitize...")
        try:
            # Fix improperly escaped quotes (e.g., \\" -> ")
            sanitized_text = response_text.replace('\\"', '"')

            # Fix improperly escaped backslashes (e.g., \\ -> \)
            sanitized_text = sanitized_text.replace('\\\\', '\\')

            # Fix trailing backslashes (e.g., +\ -> +)
            sanitized_text = sanitized_text.replace('\\\n', '\n')

            # Attempt to parse the sanitized text to ensure it is valid JSON
            json.loads(sanitized_text)  # This will raise an error if the JSON is still invalid
            return sanitized_text
        except json.JSONDecodeError as e:
            logging.error(f"Sanitization failed: {str(e)}")
            logging.debug(f"Sanitized text causing error: {sanitized_text}")
            return None

def parse_files(response_text):
    try:
        # --- FIX: Use a simple, robust JSON extraction (no (?R) regex) ---
        import re
        # Find the first top-level JSON object (from first { to last })
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            logging.error("No JSON object found in the response text.")
            logging.debug(f"Response text received: {response_text}")
            return None

        json_response = response_text[first_brace:last_brace+1]

        # Sanitize the JSON response
        sanitized_response = sanitize_response(json_response)
        if not sanitized_response:
            logging.error("Failed to sanitize the response text.")
            return None

        # Parse the sanitized JSON
        parsed_files = json.loads(sanitized_response)

        # Flatten nested dictionaries into a single-level dictionary
        def flatten_files(files_dict, parent_key=""):
            flat_files = {}
            for k, v in files_dict.items():
                new_key = f"{parent_key}/{k}" if parent_key else k
                if isinstance(v, dict):
                    flat_files.update(flatten_files(v, new_key))
                else:
                    flat_files[new_key] = v
            return flat_files

        flattened_files = flatten_files(parsed_files)

        def convert_to_string(value):
            if isinstance(value, dict):
                return json.dumps(value, indent=2)
            elif isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            return str(value)

        valid_files = {}
        for k, v in flattened_files.items():
            if not k.endswith('/') and isinstance(v, str) and v.strip():
                valid_files[k] = convert_to_string(v)
            else:
                logging.warning(f"Skipping invalid or empty entry: {k}")

        if isinstance(valid_files, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in valid_files.items()):
            return valid_files
        else:
            logging.error("Parsed response is not a valid dictionary of files.")
            logging.debug(f"Parsed response structure: {valid_files}")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {str(e)}")
        logging.debug(f"Response text causing error: {response_text}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while parsing response text: {str(e)}")
        logging.debug(f"Response text causing error: {response_text}")
        return None

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        if not data:
            logging.error("No data received in the request.")
            return jsonify({'error': 'No data received'}), 400

        prompt = data.get('prompt', '').lower()
        output_type = data.get('type', '').lower()

        logging.debug(f"Prompt: {prompt}, Type: {output_type}")

        # --- Serve local calculator template if requested ---
        if output_type == 'mobile app' and 'calculator' in prompt:
            import pathlib
            template_dir = pathlib.Path(__file__).parent / 'apk template'
            files_dict = {}
            for path in template_dir.rglob('*'):
                if path.is_file():
                    rel_path = str(path.relative_to(template_dir)).replace('\\', '/').replace(' ', '')
                    with open(path, 'rb') as f:
                        try:
                            content = f.read().decode('utf-8')
                        except Exception:
                            content = ''  # For binary files (icons), just leave empty
                    files_dict[rel_path] = content
            explanation = 'This is a prebuilt, reliable Calculator Android Studio project template.'
            return jsonify({'files': files_dict, 'explanation': explanation}), 200

        if not prompt or not output_type:
            logging.error(f"Missing prompt or type. Prompt: {prompt}, Type: {output_type}")
            return jsonify({'error': 'Prompt and type are required'}), 400

        # Use the new, softer prompt
        final_prompt = build_prompt(prompt, output_type.lower())

        logging.debug(f"Final prompt sent to OpenAI: {final_prompt}")

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 for better large project generation
            messages=[
                {"role": "system", "content": "You are a developer who returns only code files in JSON format with correct filenames and full file content. If you want to provide an explanation or summary, put it in a README.md file."},
                {"role": "user", "content": final_prompt}
            ]
        )

        # Extract the JSON object from the response
        response_text = response['choices'][0]['message']['content']
        logging.debug(f"Raw response from OpenAI: {response_text}")  # Log the raw response for debugging

        # --- Robust JSON extraction: find first { and last } ---
        cleaned = response_text.strip()
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace+1]
        else:
            logging.error("Could not find JSON object in response.")
            return jsonify({'error': 'Failed to extract JSON object from response.'}), 500

        # Try to sanitize common LLM issues (unescaped quotes, etc)
        try:
            files = json.loads(cleaned)
        except Exception as e:
            # Try to fix unescaped quotes in XML/Java values
            import re
            def fix_quotes(s):
                # Replace unescaped double quotes inside values with single quotes
                def replacer(match):
                    inner = match.group(2)
                    fixed = inner.replace('"', "'")
                    return f'{match.group(1)}"{fixed}"'
                return re.sub(r'(:\s*")([^\"]*?)(")', replacer, s)
            try:
                fixed_cleaned = fix_quotes(cleaned)
                files = json.loads(fixed_cleaned)
            except Exception as e2:
                logging.error(f"Failed to sanitize the response text: {str(e2)}")
                logging.debug(f"Sanitized text causing error: {cleaned}")
                return jsonify({'error': 'Failed to generate files. The response was incomplete or invalid.'}), 500

        # Extract README/explanation if present
        explanation = None
        readme_keys = [k for k in files if k.lower() in ['readme.md', 'readme.txt', 'readme', 'readme.markdown']]
        if readme_keys:
            explanation = files[readme_keys[0]]
            del files[readme_keys[0]]
        else:
            # Try to extract README from nested keys (sometimes LLMs nest files)
            for k in list(files.keys()):
                if 'readme' in k.lower():
                    explanation = files[k]
                    del files[k]
                    break

        # --- If still no README, try to extract from any .md file ---
        if not explanation:
            for k in list(files.keys()):
                if k.lower().endswith('.md'):
                    explanation = files[k]
                    del files[k]
                    break

        # --- If still no README, try to extract from the original response (fallback) ---
        if not explanation:
            # Try to extract README.md content from the raw response (if LLM failed to put it in JSON)
            import re
            match = re.search(r'"README\.md"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text, re.DOTALL)
            if match:
                explanation = match.group(1).encode('utf-8').decode('unicode_escape')

        logging.debug(f"Generated files: {files.keys()}")
        return jsonify({'files': files, 'explanation': explanation or "No README.md was generated."}), 200

    except Exception as e:
        logging.error(f"Error in /generate: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Let the AI decide when to generate code or just chat
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant for software development. "
                        "If the user asks for code, generate the code and explain it. "
                        "If the user is chatting, respond conversationally. "
                        "You can generate code, explain code, or just chat as appropriate."
                    )
                },
                {"role": "user", "content": message}
            ]
        )
        reply = chat_response['choices'][0]['message']['content']
        return jsonify({'message': reply}), 200

    except Exception as e:
        logging.error(f"Error in /chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-stream', methods=['POST'])
def generate_stream():
    try:
        data = request.get_json()
        if not data:
            logging.error("No data received in the request.")
            return jsonify({'error': 'No data received'}), 400

        prompt = data.get('prompt')
        output_type = data.get('type')

        if not prompt or not output_type:
            logging.error("Missing prompt or type in the request.")
            return jsonify({'error': 'Prompt and type are required'}), 400

        logging.info(f"Received prompt: {prompt}")
        logging.info(f"Received app type: {output_type}")

        final_prompt = build_prompt(prompt, output_type.lower())
        logging.debug(f"Final prompt sent to OpenAI: {final_prompt}")

        def stream_files():
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a developer who returns only code files in JSON format with correct filenames and full file content."},
                        {"role": "user", "content": final_prompt}
                    ],
                    stream=True
                )
                buffer = ""
                for chunk in response:
                    content = chunk['choices'][0]['delta'].get('content', '')
                    buffer += content
                    if buffer.endswith("}"):  # Check if the JSON object is complete
                        files = parse_files(buffer)
                        if files:
                            logging.info("Streaming files to frontend.")
                            yield f"data: {json.dumps(files)}\n\n"
                            buffer = ""  # Reset buffer after sending
            except Exception as e:
                logging.error(f"Error during streaming: {str(e)}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return Response(stream_files(), mimetype='text/event-stream')

    except Exception as e:
        logging.error(f"Error during generation process: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/sync-file', methods=['POST'])
def sync_file():
    try:
        data = request.get_json()
        filename = data.get('filename')
        content = data.get('content')

        if not filename or content is None:
            return jsonify({'error': 'Filename and content are required'}), 400

        files[filename] = content  # Update the file content in memory
        return jsonify({'message': 'File synced successfully'}), 200
    except Exception as e:
        logging.error(f"Error syncing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/file-updates', methods=['GET'])
def file_updates():
    def stream_files():
        previous_files = {}
        while True:
            if files != previous_files:
                yield f"data: {json.dumps(files)}\n\n"
                previous_files = files.copy()
    return Response(stream_with_context(stream_files()), mimetype='text/event-stream')

@app.route('/export', methods=['POST'])
def export():
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            logging.error("No valid file data received for export.")
            return jsonify({'error': 'No valid file data received'}), 400

        if not data:
            logging.error("No files provided for export.")
            return jsonify({'error': 'No files provided'}), 400

        # Create a zip file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in data.items():
                zip_file.writestr(filename, content)

        zip_buffer.seek(0)

        # Serve the zip file
        return Response(
            zip_buffer,
            mimetype='application/zip',
            headers={
                'Content-Disposition': 'attachment; filename=fronti_project.zip'
            }
        )
    except Exception as e:
        logging.error(f"Error during export: {str(e)}")
        return jsonify({'error': 'Failed to export files', 'details': str(e)}), 500

@app.route('/deploy', methods=['POST'])
def deploy():
    try:
        data = request.get_json()
        logging.debug(f"Received deployment data: {data}")  # Debugging log

        if not data or not isinstance(data, dict):
            logging.error("No valid file data received for deployment.")
            return jsonify({'error': 'No valid file data received'}), 400

        # Prepare files for deployment
        files = data.get('files', {})
        if not files:
            logging.error("No files provided for deployment.")
            return jsonify({'error': 'No files provided'}), 400

        logging.debug(f"Files to deploy: {files.keys()}")  # Debugging log

        # Create the payload for Vercel deployment
        deployment_payload = {
            "name": "fronti-web-app",
            "files": [{"file": filename, "data": content} for filename, content in files.items()],
            "projectSettings": {
                "framework": None  # Set to None for static deployments
            }
        }

        # Send the deployment request to Vercel
        headers = {
            "Authorization": f"Bearer {VERCEL_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            "https://api.vercel.com/v13/deployments",
            headers=headers,
            json=deployment_payload
        )

        logging.debug(f"Vercel API response: {response.text}")  # Debugging log

        if response.status_code != 200:
            logging.error(f"Vercel API error: {response.text}")
            return jsonify({'error': 'Failed to deploy to Vercel', 'details': response.text}), 500

        deployment_data = response.json()
        deployment_url = deployment_data.get("url")
        if not deployment_url:
            logging.error("No deployment URL returned by Vercel.")
            return jsonify({'error': 'Deployment failed. No URL returned.'}), 500

        logging.info(f"Deployment successful: {deployment_url}")
        return jsonify({'message': 'Deployment successful', 'url': f"https://{deployment_url}"})

    except Exception as e:
        logging.error(f"Error during deployment: {str(e)}")
        return jsonify({'error': 'Failed to deploy files', 'details': str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        app_type = data.get('type', '').lower()
        files = data.get('files', {})

        if not app_type or not files:
            return jsonify({'error': 'App type and files are required'}), 400

        # For web apps, just return the files as-is
        if app_type == "web app":
            # Ensure there's at least one HTML file
            html_files = [f for f in files.keys() if f.lower().endswith('.html')]
            if not html_files:
                # Create a minimal index.html if none exists
                files['index.html'] = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Generated App</title>
                </head>
                <body>
                    <h1>No HTML file was generated</h1>
                    <p>This web app doesn't contain an HTML file.</p>
                </body>
                </html>
                """
            return jsonify({'files': files})

        # For mobile/desktop apps, generate a simulated preview
        combined_files = "\n\n".join([f"### {filename}\n{content}" for filename, content in files.items()])
        conversion_prompt = f"""
        Convert the following {app_type} app project files into a single, self-contained HTML file that simulates the app.
        - Include all necessary CSS and JavaScript inline.
        - Replicate the app's behavior, interactions, and appearance as closely as possible.
        - If the app uses platform-specific features, simulate them using web technologies.
        - Ensure the generated HTML is fully functional and visually identical to the original app.

        Return ONLY the complete HTML code wrapped in <html> tags, no explanations:
        {combined_files}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You convert apps to web previews. Return only HTML code."},
                {"role": "user", "content": conversion_prompt}
            ]
        )

        converted_code = response.choices[0].message.content.strip()

        # Clean the response
        if converted_code.startswith("```html"):
            converted_code = converted_code[7:]
        if converted_code.endswith("```"):
            converted_code = converted_code[:-3]
        converted_code = converted_code.strip()

        # Validate we have proper HTML
        if not (converted_code.startswith("<!DOCTYPE html>") or "<html" in converted_code):
            # Create fallback preview
            converted_code = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{app_type} Preview</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        padding: 20px;
                        background: #f5f5f5;
                    }}
                    .container {{ 
                        max-width: 800px; 
                        margin: 0 auto; 
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{app_type.capitalize()} Preview</h1>
                    <p>This is a simulated preview of your {app_type}.</p>
                    <h3>Project Files:</h3>
                    <ul>
                        {"".join(f"<li>{f}</li>" for f in files.keys())}
                    </ul>
                </div>
            </body>
            </html>
            """

        return jsonify({'files': {'preview.html': converted_code}})

    except Exception as e:
        logging.error(f"Error in /preview: {str(e)}")
        return jsonify({'error': str(e)}), 500

def build_prompt(user_prompt, type_):
    if type_ == "desktop app":
        return f"""
You are a world-class code generator.

Generate a complete, production-ready desktop app based on this user description: "{user_prompt}"

👉 Use Python with a desktop GUI framework like Tkinter or PyQt. Alternatively, you can use Electron with JavaScript if specified in the user description.

👉 Output all files required to run this app in a single JSON object with the format:
{{ "filename.ext": "file content here", ... }}

🧠 Ensure the code is:
- Fully functional and error-free.
- Well-structured, modular, and follows best practices.
- Includes comments explaining key parts of the code.

📦 Include all necessary files, such as configuration files, dependencies, and assets.

🎯 Return only the JSON object with code files. No explanations, no markdown.
"""

    elif type_ == "mobile app":
        # SOFTER PROMPT: Less strict, more likely to succeed
        return (
            "You are a code generator.\n\n"
            f"Please generate a minimal, buildable native Android Studio project (Java, Gradle) for this app: \"{user_prompt}\"\n\n"
            "- Only native Android (no Flutter, React Native, etc).\n"
            "- Include as many required files as possible: build.gradle, settings.gradle, gradlew, app/build.gradle, app/src/main/AndroidManifest.xml, app/src/main/java/...\n"
            "- Return a JSON object: { 'filename': 'file content', ... }\n"
            "- No explanations, markdown, or extra text."
        )

    else:  # Default to web app
        return f"""
You are a world-class code generator.

Generate a complete, production-ready {type_} based on this user description: "{user_prompt}"

👉 Output all files required to run this app in a single JSON object with the format:
{{ "filename.ext": "file content here", ... }}

🧠 Ensure the code is:
- Fully functional and error-free.
- Well-structured, modular, and follows best practices.
- Includes comments explaining key parts of the code.

📦 Include all necessary files, such as configuration files, dependencies, and assets.

🎯 Return only the JSON object with code files. No explanations, no markdown.
"""

@app.route('/build-apk', methods=['POST'])
def build_apk():
    """
    Accepts mobile app project files, builds an APK (Flutter, React Native, or native Android), and returns the APK file.
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'No valid file data received'}), 400
        files = data.get('files', {})
        if not files:
            return jsonify({'error': 'No files provided'}), 400

        # Create a temporary directory for the build
        temp_dir = tempfile.mkdtemp(prefix="apkbuild_")
        try:
            # Write files to temp_dir
            for filename, content in files.items():
                file_path = os.path.join(temp_dir, filename.replace("..", ""))
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            # --- PATCH: Inject standard gradlew, gradlew.bat, gradle-wrapper if missing ---
            gradlew_path = os.path.join(temp_dir, 'gradlew')
            gradlewb_path = os.path.join(temp_dir, 'gradlew.bat')
            wrapper_dir = os.path.join(temp_dir, 'gradle', 'wrapper')
            wrapper_jar = os.path.join(wrapper_dir, 'gradle-wrapper.jar')
            wrapper_props = os.path.join(wrapper_dir, 'gradle-wrapper.properties')
            # Add gradlew (Unix shell script)
            if not os.path.exists(gradlew_path):
                with open(gradlew_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(_GRADLEW_SCRIPT)
                os.chmod(gradlew_path, 0o755)
            # Add gradlew.bat (Windows batch script)
            if not os.path.exists(gradlewb_path):
                with open(gradlewb_path, 'w', encoding='utf-8', newline='\r\n') as f:
                    f.write(_GRADLEW_BAT)
            # Add gradle-wrapper.properties
            if not os.path.exists(wrapper_dir):
                os.makedirs(wrapper_dir, exist_ok=True)
            if not os.path.exists(wrapper_props):
                with open(wrapper_props, 'w', encoding='utf-8') as f:
                    f.write(_GRADLE_WRAPPER_PROPERTIES)
            # Add gradle-wrapper.jar (empty placeholder, real build will fail if missing, but at least structure is there)
            if not os.path.exists(wrapper_jar):
                with open(wrapper_jar, 'wb') as f:
                    f.write(b'')
            # Ensure required folders exist
            for folder in [
                os.path.join(temp_dir, 'app', 'src', 'main', 'java'),
                os.path.join(temp_dir, 'app', 'src', 'main', 'res'),
                os.path.join(temp_dir, 'app', 'src', 'main', 'res', 'layout')
            ]:
                os.makedirs(folder, exist_ok=True)

            # --- PATCH: Inject fallback layout and logic if missing ---
            layout_path = os.path.join(temp_dir, 'app', 'src', 'main', 'res', 'layout', 'activity_main.xml')
            main_activity_path = os.path.join(temp_dir, 'app', 'src', 'main', 'java', 'com', 'example', 'calculator', 'MainActivity.java')
            injected_warning = None
            if not os.path.exists(layout_path):
                with open(layout_path, 'w', encoding='utf-8') as f:
                    f.write('''<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <EditText
        android:id="@+id/display"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:inputType="none"
        android:focusable="false"
        android:textSize="32sp"
        android:gravity="end"
        android:background="#EEE"
        android:padding="12dp" />

    <GridLayout
        android:id="@+id/buttonGrid"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:columnCount="4"
        android:rowCount="5"
        android:layout_marginTop="16dp">

        <!-- Calculator buttons -->
        <Button android:id="@+id/btn7" android:text="7" />
        <Button android:id="@+id/btn8" android:text="8" />
        <Button android:id="@+id/btn9" android:text="9" />
        <Button android:id="@+id/btnDiv" android:text="/" />

        <Button android:id="@+id/btn4" android:text="4" />
        <Button android:id="@+id/btn5" android:text="5" />
        <Button android:id="@+id/btn6" android:text="6" />
        <Button android:id="@+id/btnMul" android:text="*" />

        <Button android:id="@+id/btn1" android:text="1" />
        <Button android:id="@+id/btn2" android:text="2" />
        <Button android:id="@+id/btn3" android:text="3" />
        <Button android:id="@+id/btnSub" android:text="-" />

        <Button android:id="@+id/btn0" android:text="0" />
        <Button android:id="@+id/btnDot" android:text="." />
        <Button android:id="@+id/btnEq" android:text="=" />
        <Button android:id="@+id/btnAdd" android:text="+" />

        <Button android:id="@+id/btnClear" android:text="C" />
        <Button android:id="@+id/btnParenL" android:text="(" />
        <Button android:id="@+id/btnParenR" android:text=")" />
        <Button android:id="@+id/btnPerc" android:text="%" />
    </GridLayout>
</LinearLayout>
''')
                injected_warning = "activity_main.xml was missing and has been auto-generated."
            # Patch MainActivity.java if missing or if it only contains the default stub
            if not os.path.exists(main_activity_path):
                os.makedirs(os.path.dirname(main_activity_path), exist_ok=True)
                with open(main_activity_path, 'w', encoding='utf-8') as f:
                    f.write('''package com.example.calculator;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.view.View;

public class MainActivity extends AppCompatActivity {
    private EditText display;
    private StringBuilder input = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        display = findViewById(R.id.display);
        int[] btnIds = {R.id.btn0, R.id.btn1, R.id.btn2, R.id.btn3, R.id.btn4, R.id.btn5, R.id.btn6, R.id.btn7, R.id.btn8, R.id.btn9,
                R.id.btnAdd, R.id.btnSub, R.id.btnMul, R.id.btnDiv, R.id.btnDot, R.id.btnParenL, R.id.btnParenR, R.id.btnPerc};
        String[] btnVals = {"0","1","2","3","4","5","6","7","8","9",
                "+","-","*","/",".","(",")","%"};
        for (int i = 0; i < btnIds.length; i++) {
            Button b = findViewById(btnIds[i]);
            String val = btnVals[i];
            b.setOnClickListener(v -> {
                input.append(val);
                display.setText(input.toString());
            });
        }
        Button btnEq = findViewById(R.id.btnEq);
        btnEq.setOnClickListener(v -> {
            try {
                double result = eval(input.toString());
                display.setText(Double.toString(result));
                input.setLength(0);
            } catch (Exception e) {
                display.setText("Error");
                input.setLength(0);
            }
        });
        Button btnClear = findViewById(R.id.btnClear);
        btnClear.setOnClickListener(v -> {
            input.setLength(0);
            display.setText("");
        });
    }
    // Simple eval (WARNING: for demo only, not safe for production)
    private double eval(String expr) {
        return new Object() {
            int pos = -1, ch;
            void nextChar() { ch = (++pos < expr.length()) ? expr.charAt(pos) : -1; }
            boolean eat(int charToEat) {
                while (ch == ' ') nextChar();
                if (ch == charToEat) { nextChar(); return true; }
                return false;
            }
            double parse() {
                nextChar();
                double x = parseExpression();
                if (pos < expr.length()) throw new RuntimeException("Unexpected: " + (char)ch);
                return x;
            }
            double parseExpression() {
                double x = parseTerm();
                for (;;) {
                    if      (eat('+')) x += parseTerm();
                    else if (eat('-')) x -= parseTerm();
                    else return x;
                }
            }
            double parseTerm() {
                double x = parseFactor();
                for (;;) {
                    if      (eat('*')) x *= parseFactor();
                    else if (eat('/')) x /= parseFactor();
                    else if (eat('%')) x %= parseFactor();
                    else return x;
                }
            }
            double parseFactor() {
                if (eat('+')) return parseFactor();
                if (eat('-')) return -parseFactor();
                double x;
                int startPos = this.pos;
                if (eat('(')) {
                    x = parseExpression();
                    eat(')');
                } else if ((ch >= '0' && ch <= '9') || ch == '.') {
                    while ((ch >= '0' && ch <= '9') || ch == '.') nextChar();
                    x = Double.parseDouble(expr.substring(startPos, this.pos));
                } else {
                    throw new RuntimeException("Unexpected: " + (char)ch);
                }
                return x;
            }
        }.parse();
    }
}
''')
                injected_warning = (injected_warning or "") + " MainActivity.java was missing and has been auto-generated with basic calculator logic."

            # Detect framework
            framework = None
            if os.path.exists(os.path.join(temp_dir, 'pubspec.yaml')):
                framework = 'flutter'
            elif os.path.exists(os.path.join(temp_dir, 'android', 'build.gradle')) or os.path.exists(os.path.join(temp_dir, 'build.gradle')):
                framework = 'native'
            elif os.path.exists(os.path.join(temp_dir, 'package.json')):
                with open(os.path.join(temp_dir, 'package.json'), 'r', encoding='utf-8') as f:
                    pkg = json.load(f)
                    deps = pkg.get('dependencies', {})
                    if 'react-native' in deps:
                        framework = 'react-native'

            if not framework:
                shutil.rmtree(temp_dir)
                return jsonify({'error': 'Could not detect mobile framework (Flutter, React Native, or native Android)'}), 400

            # --- Validate required files/folders for mobile frameworks ---
            # Only allow native Android (no Flutter, no React Native)
            missing = []
            # Must have build.gradle (project), app/build.gradle, settings.gradle, gradlew or gradlew.bat, app/src/main/AndroidManifest.xml
            if not os.path.exists(os.path.join(temp_dir, 'build.gradle')):
                missing.append('build.gradle (project root)')
            if not os.path.exists(os.path.join(temp_dir, 'settings.gradle')):
                missing.append('settings.gradle')
            if not os.path.exists(os.path.join(temp_dir, 'gradlew')) and not os.path.exists(os.path.join(temp_dir, 'gradlew.bat')):
                missing.append('gradlew or gradlew.bat')
            if not os.path.exists(os.path.join(temp_dir, 'app', 'build.gradle')):
                missing.append('app/build.gradle')
            if not os.path.exists(os.path.join(temp_dir, 'app', 'src', 'main', 'AndroidManifest.xml')):
                missing.append('app/src/main/AndroidManifest.xml')
            if not os.path.exists(os.path.join(temp_dir, 'app', 'src', 'main', 'java')):
                missing.append('app/src/main/java (Java source folder)')
            if missing:
                shutil.rmtree(temp_dir)
                return jsonify({'error': f"The generated project is missing required files/folders for a native Android (Gradle/Java) app: {', '.join(missing)}. Please regenerate and ensure a real Android Studio project is created."}), 400

            apk_path = None
            build_log = ""
            if framework == 'flutter':
                # Flutter build
                build_cmd = ['flutter', 'build', 'apk', '--release']
                build_proc = subprocess.run(build_cmd, cwd=temp_dir, capture_output=True, text=True)
                build_log = build_proc.stdout + "\n" + build_proc.stderr
                if build_proc.returncode != 0:
                    shutil.rmtree(temp_dir)
                    return jsonify({'error': 'Flutter build failed', 'log': build_log}), 500
                # Find APK
                apk_dir = os.path.join(temp_dir, 'build', 'app', 'outputs', 'flutter-apk')
                for fname in os.listdir(apk_dir):
                    if fname.endswith('.apk'):
                        apk_path = os.path.join(apk_dir, fname)
                        break
            elif framework == 'react-native':
                # React Native build (assumes Android)
                # Install dependencies
                subprocess.run(['npm', 'install'], cwd=temp_dir)
                # Build APK
                gradlew = 'gradlew.bat' if os.name == 'nt' else './gradlew'
                android_dir = os.path.join(temp_dir, 'android')
                build_cmd = [gradlew, 'assembleRelease']
                build_proc = subprocess.run(build_cmd, cwd=android_dir, capture_output=True, text=True, shell=True)
                build_log = build_proc.stdout + "\n" + build_proc.stderr
                if build_proc.returncode != 0:
                    shutil.rmtree(temp_dir)
                    return jsonify({'error': 'React Native build failed', 'log': build_log}), 500
                # Find APK
                apk_dir = os.path.join(android_dir, 'app', 'build', 'outputs', 'apk', 'release')
                for fname in os.listdir(apk_dir):
                    if fname.endswith('.apk'):
                        apk_path = os.path.join(apk_dir, fname)
                        break
            elif framework == 'native':
                # Native Android build
                gradlew = 'gradlew.bat' if os.name == 'nt' else './gradlew'
                build_cmd = [gradlew, 'assembleRelease']
                build_proc = subprocess.run(build_cmd, cwd=temp_dir, capture_output=True, text=True, shell=True)
                build_log = build_proc.stdout + "\n" + build_proc.stderr
                if build_proc.returncode != 0:
                    shutil.rmtree(temp_dir)
                    return jsonify({'error': 'Native Android build failed', 'log': build_log}), 500
                # Find APK
                apk_dir = os.path.join(temp_dir, 'app', 'build', 'outputs', 'apk', 'release')
                for fname in os.listdir(apk_dir):
                    if fname.endswith('.apk'):
                        apk_path = os.path.join(apk_dir, fname)
                        break
            else:
                shutil.rmtree(temp_dir)
                return jsonify({'error': 'Unsupported framework'}), 400

            if not apk_path or not os.path.exists(apk_path):
                shutil.rmtree(temp_dir)
                return jsonify({'error': 'APK not found after build', 'log': build_log}), 500

            # Return APK file
            apk_filename = os.path.basename(apk_path)
            with open(apk_path, 'rb') as apk_file:
                apk_bytes = apk_file.read()
            shutil.rmtree(temp_dir)
            headers = {
                'Content-Disposition': f'attachment; filename={apk_filename}'
            }
            if injected_warning:
                headers['X-Fronti-Warning'] = injected_warning
            return Response(
                apk_bytes,
                mimetype='application/vnd.android.package-archive',
                headers=headers
            )
        except Exception as e:
            shutil.rmtree(temp_dir)
            logging.error(f"Error during APK build: {str(e)}")
            return jsonify({'error': 'Build process failed', 'details': str(e)}), 500
    except Exception as e:
        logging.error(f"Error in /build-apk: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-structure', methods=['POST'])
def generate_structure():
    """
    Step 1: Get a complete file/folder list for a minimal, buildable Android Studio project.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        structure_prompt = (
            "You are a senior Android developer. "
            f"Given this app description: '{prompt}', "
            "list ALL files and folders (with full relative paths) required for a minimal, buildable native Android Studio project (Java, Gradle). "
            "Do NOT include any Flutter, React Native, or cross-platform files. "
            "Return ONLY a JSON array of file and folder paths, e.g. ['build.gradle', 'settings.gradle', 'app/build.gradle', 'app/src/main/AndroidManifest.xml', ...]. "
            "No explanations, no markdown, no extra text."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a developer who returns only JSON arrays of file/folder paths."},
                {"role": "user", "content": structure_prompt}
            ]
        )
        response_text = response['choices'][0]['message']['content']
        # Remove code block markers if present
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        # Try to parse as JSON array
        import json
        try:
            structure = json.loads(cleaned)
            if isinstance(structure, list):
                return jsonify({'structure': structure}), 200
            else:
                return jsonify({'error': 'Did not receive a JSON array.'}), 500
        except Exception as e:
            return jsonify({'error': f'Failed to parse structure: {str(e)}', 'raw': cleaned}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-file', methods=['POST'])
def generate_file():
    """
    Step 2: Get the content for a specific file in the Android project.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        filename = data.get('filename')
        if not prompt or not filename:
            return jsonify({'error': 'Prompt and filename are required'}), 400
        file_prompt = (
            "You are a senior Android developer. "
            f"Given this app description: '{prompt}', "
            f"generate the FULL content for this file in a minimal, buildable native Android Studio project (Java, Gradle): {filename}\n"
            "Return ONLY the file content as plain text. No explanations, no markdown, no code block, no extra text."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a developer who returns only file content as plain text."},
                {"role": "user", "content": file_prompt}
            ]
        )
        file_content = response['choices'][0]['message']['content'].strip()
        # Remove code block markers if present
        if file_content.startswith("```java"):
            file_content = file_content[7:]
        if file_content.startswith("```xml"):
            file_content = file_content[6:]
        if file_content.startswith("```gradle"):
            file_content = file_content[9:]
        if file_content.startswith("```"):
            file_content = file_content[3:]
        if file_content.endswith("```"):
            file_content = file_content[:-3]
        file_content = file_content.strip()
        return jsonify({'filename': filename, 'content': file_content}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Standard gradle wrapper scripts and properties ---
_GRADLEW_SCRIPT = """#!/usr/bin/env sh\n\n##############################################################################\n## Gradle start up script for UN*X\n##############################################################################\n\n# (This is a standard gradlew script. For real builds, copy from a real Android Studio project.)\n\nif [ -z "$JAVA_HOME" ]; then\n  JAVA_CMD=java\nelse\n  JAVA_CMD="$JAVA_HOME/bin/java"\nfi\n\nCLASSPATH=""\nfor jar in \"$(dirname "$0")/gradle/wrapper/gradle-wrapper.jar\"; do\n  CLASSPATH=\"$CLASSPATH:$jar\"\ndone\n\nexec "$JAVA_CMD" -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"\n"""
_GRADLEW_BAT = """@echo off\n\nREM -----------------------------------------------------------------------------\nREM Gradle start up script for Windows\nREM -----------------------------------------------------------------------------\n\nSETLOCAL\nSET DIRNAME=%~dp0\nSET APP_BASE_NAME=%~n0\nSET CLASSPATH=%DIRNAME%gradle\wrapper\gradle-wrapper.jar\n\nSET JAVA_EXE=java\nIF NOT "%JAVA_HOME%"=="" SET JAVA_EXE=%JAVA_HOME%\bin\java\n\n%JAVA_EXE% -classpath %CLASSPATH% org.gradle.wrapper.GradleWrapperMain %*\n"""
_GRADLE_WRAPPER_PROPERTIES = """distributionUrl=https\://services.gradle.org/distributions/gradle-7.6.4-all.zip\n"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)