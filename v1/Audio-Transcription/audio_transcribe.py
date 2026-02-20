#!/usr/bin/env python3
"""
Optimized Audio Transcription Script (No Diarization)
Supports command-line arguments for CPU threads and output file name.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import whisperx
import torch
from dotenv import load_dotenv


class AudioTranscriber:
    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", num_threads: int = 2):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_threads = num_threads
        self.model = None

        torch.set_num_threads(num_threads)
        load_dotenv()  # Load .env file if available

    def load_model(self) -> None:
        if self.model is None:
            print(f"Loading {self.model_size} Whisper model on {self.device}...")
            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print("✅ Whisper model loaded successfully!")

    def transcribe_audio(self, audio_path: str, language: str = "en") -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.load_model()

        print(f"🎵 Transcribing: {audio_path}")
        start_time = time.time()

        result = self.model.transcribe(
            audio_path,
            language=language,
            batch_size=8,  # adjust if memory issues
        )

        duration = time.time() - start_time
        print(f"✅ Transcription completed in {duration:.2f} seconds")
        return result

    def save_transcription(self, result: dict, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"💾 Transcription saved to: {output_path}")

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        import gc
        gc.collect()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio Transcription (No Diarization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python audio_transcribe.py input.mp3 -o output.json -t 4
  python audio_transcribe.py input.wav -o results/transcript.json -t 2 -m base
"""
    )

    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument("-o", "--output", help="Output file path", type=str)
    parser.add_argument("-t", "--threads", help="Number of CPU threads", type=int, default=2)
    parser.add_argument("-m", "--model", choices=["tiny", "base", "small", "medium", "large"], default="small")
    parser.add_argument("-l", "--language", help="Language code", default="en")
    parser.add_argument("-d", "--device", choices=["cpu"], default="cpu")
    return parser.parse_args()


def generate_output_path(audio_path: str, custom_output: Optional[str] = None) -> str:
    if custom_output:
        return custom_output

    audio_file = Path(audio_path)
    output_dir = "output"
    output_file = f"{audio_file.stem}_transcript.json"
    return os.path.join(output_dir, output_file)


def main():
    try:
        args = parse_arguments()
        output_path = generate_output_path(args.audio_file, args.output)

        print("=" * 70)
        print("🎤 Audio Transcription (No Diarization)")
        print("=" * 70)
        print(f"📁 Input file: {args.audio_file}")
        print(f"📄 Output file: {output_path}")
        print(f"🧵 CPU threads: {args.threads}")
        print(f"🤖 Model: {args.model}")
        print(f"🌐 Language: {args.language}")
        print(f"💻 Device: {args.device}")
        print("=" * 70)

        transcriber = AudioTranscriber(
            model_size=args.model,
            device=args.device,
            num_threads=args.threads
        )

        transcription_result = transcriber.transcribe_audio(args.audio_file, args.language)

        transcription_result["metadata"] = {
            "audio_file": args.audio_file,
            "model_size": args.model,
            "language": args.language,
            "device": args.device,
            "processing_time": time.time()
        }

        transcriber.save_transcription(transcription_result, output_path)

        transcriber.cleanup()

        print("\n🎉 Transcription completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Transcription interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
