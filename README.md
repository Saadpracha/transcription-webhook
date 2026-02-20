## Leadgeneration Webhook – Multi‑Version Deployment

This repository contains the Webhook API codebase and assets used for a multi‑version FastAPI deployment behind `webhook.lead2424.com`.

### Project Structure (High‑Level)

- **v1/**
  - **Audio-Transcription/**
    - Application code (`fastapi_whisper_service.py`, `audio_transcribe.py`, `audio_transcribe_diarization.py`, `ai_summary.py`)
    - App assets (`templates/`, `static/`)
    - Configuration & metadata (`requirements.txt`, `prompt.json`, `README.md`)

- **v2/**
  - **Audio-Transcription/**
    - Application code (`fastapi_whisper_service.py`, `audio_transcribe.py`, `audio_transcribe_diarization.py`, `ai_summary.py`)
    - App assets (`templates/`, `static/`)
    - Configuration & metadata (`requirements.txt`, `prompt.json`, `README.md`)

- **v3/**
  - **Audio-Transcription/**
    - Application code (`fastapi_whisper_service.py`, `audio_transcribe.py`, `audio_transcribe_diarization.py`, `ai_summary.py`)
    - App assets (`templates/`, `static/`)
    - Configuration & metadata (`requirements.txt`, `prompt.json`, `README.md`)

- **.gitignore**

### Server Deployment Summary

We successfully implemented a production‑ready multi‑version FastAPI deployment architecture on the server for the Webhook API (`webhook.lead2424.com`). Three isolated versions (`v1`, `v2`, `v3`) were deployed under `/home/saad/webhook/`, each running in its own dedicated virtual environment to prevent dependency conflicts and allow safe independent updates. Every version runs on a separate internal port (`8001`, `8002`, `8003`) and exposes the required `POST /webhook` endpoint and `GET /health` endpoint. Each version is managed through its own `systemd` service, ensuring automatic restarts, controlled resource usage, and operational stability.

To ensure production safety and long‑term reliability, we configured structured logging, log rotation, memory and CPU limits per service, and added swap space to prevent crashes under load. The deployment workflow now supports isolated updates and rollbacks per version without affecting others, maintaining strict version separation and infrastructure clarity. The system is now stable, scalable, and fully aligned with the client’s multi‑version deployment standard.

