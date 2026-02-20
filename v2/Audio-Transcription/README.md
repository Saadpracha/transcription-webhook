# Audio-Transcription
Audio transcription of GHL calls

# GHL ↔ Whisper Transcription (Prototype)

## Project Goal
Build a **simple, working prototype** that automatically transcribes **inbound and outbound** GoHighLevel (GHL) call recordings using **OpenAI Whisper**, then posts the **full transcript** back into GHL as a **Note** on the Contact (referencing the Opportunity when provided). Transcripts are stored **locally on disk** and exposed via a read-only URL so they can be linked from GHL.

## Background
- **Trigger**: A GHL Workflow with **Call Status = Completed** sends a JSON payload to our endpoint that includes the **recording URL** (via `{{message.attachments}}` or an explicit field).
- **Server**: A small **Python/FastAPI** app on **AWS Ubuntu** receives the webhook, downloads the audio, calls **OpenAI Whisper** (`audio/transcriptions`), and writes **TXT** and **SRT** files under a local directory (e.g., `/var/ghl-whisper/transcripts/`).
- **Redaction**: The transcript is auto-redacted for **emails** and **credit card numbers** (Luhn check). **Phone numbers are not redacted**.
- **GHL Update**: The app creates a **Contact Note** that includes call direction/duration, the original recording link, and links to the TXT/SRT transcript files. If a **Custom Field ID** for “Transcript URL” is provided, it also updates that field on the Contact with the TXT link.
- **Timezone**: Assumes **America/Toronto** for timestamps and filenames.
- **Deliberate constraints (MVP)**: No summaries, no diarization, **no S3** (disk only), no Make.com, and no tags/automations beyond the Note and optional custom field.

