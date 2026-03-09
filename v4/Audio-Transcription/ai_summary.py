import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key securely from an environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key not found. Please set the environment variable.")
    exit()

# Function to load your transcript from a file
def load_transcript(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to load prompt from JSON file
def load_prompt_from_json(prompt_file_path: str = "prompt.json") -> str:
    """Load prompt template from JSON file. If file doesn't exist, use default prompt."""
    try:
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r', encoding='utf-8') as file:
                prompt_data = json.load(file)
                return prompt_data.get("prompt", get_default_prompt())
        else:
            print(f"Prompt file {prompt_file_path} not found, using default prompt")
            return get_default_prompt()
    except Exception as e:
        print(f"Error loading prompt from {prompt_file_path}: {e}, using default prompt")
        return get_default_prompt()

# Function to get default prompt
def get_default_prompt() -> str:
    """Return the default prompt template."""
    return """You are an expert meeting and call analysis assistant.
Your job is to read the transcript of a sales or support call and return ONLY valid JSON.

The JSON must follow this schema exactly:
{{
  "summary": "string – a clear, concise summary of the call (cover goals, decisions, commitments, numbers, and outcomes).",
  "call_to_action_items": [
    {{
      "item": "string – the specific task or next step",
      "owner": "string – person responsible (use speaker names if clear, else leave empty)",
      "due": "string – due date if mentioned, else empty"
    }}
  ],
  "call_quality_feedback": {{
    "strengths": ["list of things done well – rapport, clarity, active listening, etc."],
    "improvements": ["list of ways caller/agent can improve – tone, pacing, missing info, objection handling, etc."]
  }}
}}

Guidelines:
- Be **brief but insightful**: a short call = short summary; a long/complex call = more detail.
- Summaries should highlight outcomes, decisions, and commitments – not just a play-by-play.
- Call-to-actions must be **specific and actionable**. Avoid vague items like "follow up" unless no detail is provided.
- Owners: infer from speaker labels where possible. Example: if "Agent" says "I will send the proposal", set owner = "Agent".
- If dates are mentioned, capture them (e.g. "by Friday"); if not, leave due = "".
- Feedback should be constructive: balance what went well with what could improve.
- Use caller information when available to provide more personalized analysis and better identify who is speaking in the transcript.

TRANSCRIPT:
<<<
{full_transcript}
>>>"""

# Function to create the API prompt
def create_prompt(transcript, prompt_file_path: str = "prompt.json", caller_first_name: str = "", caller_last_name: str = ""):
    # Support multiple transcript shapes
    if isinstance(transcript, dict) and "segments" in transcript:
        # Expecting list of {'text': ...}
        full_transcript = "\n".join([seg.get('text', '') for seg in transcript['segments']])
    elif isinstance(transcript, dict) and "text" in transcript:
        full_transcript = transcript["text"]
    else:
        # Fallback: stringify the transcript object
        full_transcript = json.dumps(transcript, ensure_ascii=False)

    # Load prompt template from JSON file
    prompt_template = load_prompt_from_json(prompt_file_path)
    
    # Add caller information to the prompt if provided
    if caller_first_name or caller_last_name:
        caller_name = f"{caller_first_name} {caller_last_name}".strip()
        # Add caller context to the prompt instructions with strict normalization guidance
        caller_context = (
            "\n\nIMPORTANT CONTEXT:\n"
            f"The caller's name is: {caller_name}\n"
            "Always refer to the caller by this exact name and spelling in all outputs. "
            "Normalize any ASR misrecognitions of the caller's name (e.g., 'Eliza', 'Aliza') to the provided name above. "
            "If a company is mentioned for the caller (e.g., ENOX Communication), include it once in the summary as '"
            f"{caller_name} from <Company>' when clear.\n\n"
        )
        # Insert caller context before the transcript section
        prompt_template = prompt_template.replace("TRANSCRIPT:\n<<<", f"{caller_context}TRANSCRIPT:\n<<<")
        
        # Also add caller information to the transcript for reference
        caller_info = f"\n\nCALLER INFORMATION:\nCaller Name: {caller_name}\n\n"
        full_transcript = caller_info + full_transcript

    # Safely inject transcript without interpreting other braces in the template
    # Only replace the specific placeholder token
    prompt = prompt_template.replace("{full_transcript}", full_transcript)
    return prompt

# Function to call OpenAI's API and get the summary
def get_summary_from_openai(prompt, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # you can also use "gpt-4o"
        messages=[
            {"role": "system", "content": "You are a helpful meeting-notes assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7,
    )

    # Return the assistant content (defensive)
    try:
        return response.choices[0].message.content.strip()
    except Exception:
        return str(response)

# Helper: strip code fences like ```json ... ```
def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text

# Save parsed JSON and then summary + call_to_action to separate files
def save_outputs(parsed_json: dict, base_output_path: str):
    # Save full parsed JSON
    with open(base_output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_json, f, indent=4, ensure_ascii=False)
    print(f"Full parsed JSON saved to {base_output_path}")

    root, _ = os.path.splitext(base_output_path)
    summary_path = f"{root}_summary.json"
    cta_path = f"{root}_call_to_action.json"

    # Extract summary and call_to_action keys (simple/flexible)
    summary_val = parsed_json.get("summary", "")
    # try common CTA keys
    cta_val = None
    for k in ("call_to_action", "call_to_action_items", "call_to_actions", "callToAction", "call_to_action_item"):
        if k in parsed_json:
            cta_val = parsed_json[k]
            break

    # Extract call quality feedback (support a few variants)
    call_quality_val = None
    for k in ("call_quality_feedback", "call_quality", "quality_feedback", "callQualityFeedback"):
        if k in parsed_json:
            call_quality_val = parsed_json[k]
            break

    # Normalize call quality feedback
    if not isinstance(call_quality_val, dict):
        call_quality_val = {"strengths": [], "improvements": []}
    else:
        if "strengths" not in call_quality_val or not isinstance(call_quality_val.get("strengths"), list):
            call_quality_val["strengths"] = list(call_quality_val.get("strengths", []))
        if "improvements" not in call_quality_val or not isinstance(call_quality_val.get("improvements"), list):
            call_quality_val["improvements"] = list(call_quality_val.get("improvements", []))

    # Save summary JSON with embedded call quality feedback and call to action
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary_val,
            "call_quality_feedback": call_quality_val,
            "call_to_action": cta_val if cta_val is not None else []
        }, f, indent=4, ensure_ascii=False)
    print(f"Summary saved to {summary_path} (includes call to action data)")

    # No longer create separate call to action file
    print("Call to action data included in summary file")

# Save raw model response when JSON parse fails
def save_raw_response(raw_text: str, base_output_path: str):
    root, _ = os.path.splitext(base_output_path)
    raw_path = f"{root}.raw.txt"
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(raw_text)
    print(f"Raw response saved to {raw_path}")

# Main function to process the transcript and generate the summary
def process_transcript(input_path, output_path, api_key, prompt_file_path="prompt.json", caller_first_name="", caller_last_name=""):
    transcript = load_transcript(input_path)
    if transcript is None:
        return

    prompt = create_prompt(transcript, prompt_file_path, caller_first_name, caller_last_name)
    response_text = get_summary_from_openai(prompt, api_key)
    if not response_text:
        print("No response from OpenAI.")
        return

    cleaned = strip_code_fences(response_text)
    try:
        parsed = json.loads(cleaned)
        save_outputs(parsed, output_path)
    except json.JSONDecodeError as e:
        print(f"Error decoding the response: {e}")
        print("Raw response:", response_text)
        save_raw_response(response_text, output_path)

# Example usage (update paths if needed)
if __name__ == "__main__":
    input_path = r"E:\work\leadgeneration\audio_transcription\Audio-Transcription\output\audio_1_diarization.json"
    output_path = r"E:\work\leadgeneration\audio_transcription\Audio-Transcription\output\summary2.json"
    process_transcript(input_path, output_path, api_key)
