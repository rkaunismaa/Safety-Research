#!/usr/bin/env python3
"""
Voice Chat with Claude

A simple voice interface for conversing with Claude using:
- faster-whisper (speech-to-text) on GPU 0
- Piper (text-to-speech) for responses
- Anthropic API for Claude

Usage:
    python voice_chat.py

Controls:
    - Hold SPACE to record your message
    - Release SPACE to send to Claude
    - Press 'q' to quit
    - Press 'c' to clear conversation history
"""

import os
import sys
import tempfile
import threading
import queue
import subprocess
from pathlib import Path

# Set CUDA device before importing torch/whisper
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from anthropic import Anthropic

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_MODEL = "large-v3"  # Options: tiny, base, small, medium, large-v2, large-v3
CLAUDE_MODEL = "claude-sonnet-4-20250514"
PIPER_VOICE = Path.home() / ".local/share/piper/en_US-lessac-medium.onnx"

# ANSI colors for terminal output
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_status(msg: str, color: str = Colors.CYAN):
    print(f"{color}{msg}{Colors.END}")


def print_user(msg: str):
    print(f"\n{Colors.GREEN}{Colors.BOLD}You:{Colors.END} {msg}")


def print_claude(msg: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}Claude:{Colors.END} {msg}")


class VoiceChat:
    def __init__(self):
        print_status("Initializing Voice Chat with Claude...", Colors.YELLOW)

        # Initialize Whisper on GPU 0
        print_status("Loading Whisper model (this may take a moment)...")
        self.whisper = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="float16"
        )
        print_status(f"Whisper {WHISPER_MODEL} loaded on GPU 0")

        # Initialize Anthropic client
        self.client = Anthropic()
        print_status("Anthropic client initialized")

        # Check Piper voice exists
        if not PIPER_VOICE.exists():
            print_status(f"Warning: Piper voice not found at {PIPER_VOICE}", Colors.RED)
            print_status("Run the setup commands from the README to download it", Colors.YELLOW)
            self.tts_enabled = False
        else:
            self.tts_enabled = True
            print_status(f"Piper voice loaded: {PIPER_VOICE.name}")

        # Conversation history
        self.messages = []

        # Audio recording state
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recorded_audio = []

        print_status("\nReady! Hold SPACE to talk, release to send. Press 'q' to quit.\n", Colors.GREEN)

    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block during recording."""
        if status:
            print_status(f"Audio status: {status}", Colors.RED)
        if self.recording:
            self.audio_queue.put(indata.copy())

    def record_audio(self) -> np.ndarray:
        """Record audio while self.recording is True."""
        self.recorded_audio = []

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self.audio_callback
        ):
            while self.recording:
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    self.recorded_audio.append(data)
                except queue.Empty:
                    continue

        if not self.recorded_audio:
            return np.array([])

        return np.concatenate(self.recorded_audio, axis=0)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper."""
        if len(audio) == 0:
            return ""

        # Whisper expects float32 audio
        audio = audio.flatten().astype(np.float32)

        segments, info = self.whisper.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True
        )

        text = " ".join(segment.text for segment in segments).strip()
        return text

    def speak(self, text: str):
        """Convert text to speech using Piper."""
        if not self.tts_enabled:
            return

        try:
            # Use piper to generate speech and play with aplay
            process = subprocess.Popen(
                [
                    "piper",
                    "--model", str(PIPER_VOICE),
                    "--output-raw"
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            # Pipe to aplay for playback
            aplay = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
                stdin=process.stdout,
                stderr=subprocess.DEVNULL
            )

            process.stdin.write(text.encode("utf-8"))
            process.stdin.close()
            aplay.wait()
            process.wait()

        except FileNotFoundError:
            print_status("Piper or aplay not found. Install with: pip install piper-tts && sudo apt install alsa-utils", Colors.RED)
            self.tts_enabled = False
        except Exception as e:
            print_status(f"TTS error: {e}", Colors.RED)

    def chat(self, user_message: str) -> str:
        """Send message to Claude and get response."""
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system="You are a helpful AI assistant engaged in a voice conversation. Keep responses concise and conversational - typically 1-3 sentences unless more detail is specifically requested. Avoid using markdown formatting, bullet points, or code blocks since your responses will be read aloud.",
            messages=self.messages
        )

        assistant_message = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        print_status("\nConversation history cleared.\n", Colors.YELLOW)

    def run(self):
        """Main loop using keyboard input."""
        try:
            import termios
            import tty

            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)

            try:
                # Set terminal to raw mode for key detection
                tty.setcbreak(sys.stdin.fileno())

                recording_thread = None

                while True:
                    # Check for keypress
                    if sys.stdin in self._select_stdin():
                        key = sys.stdin.read(1)

                        if key == 'q':
                            print_status("\nGoodbye!", Colors.YELLOW)
                            break

                        elif key == 'c':
                            self.clear_history()

                        elif key == ' ':
                            if not self.recording:
                                # Start recording
                                print_status("Recording... (release SPACE to send)", Colors.GREEN)
                                self.recording = True
                                recording_thread = threading.Thread(target=self.record_audio)
                                recording_thread.start()

                        elif key == '\x00' or ord(key) == 0:
                            # Key release (space released)
                            pass

                    # Check if space was released (recording should stop)
                    # We detect this by checking if space is no longer being held
                    if self.recording:
                        # Small delay then check if still holding
                        import select
                        if not select.select([sys.stdin], [], [], 0.05)[0]:
                            # No new keypress, stop recording
                            self.recording = False
                            if recording_thread:
                                recording_thread.join()

                            print_status("Processing...", Colors.YELLOW)

                            # Get recorded audio
                            audio = np.concatenate(self.recorded_audio) if self.recorded_audio else np.array([])

                            if len(audio) > SAMPLE_RATE * 0.5:  # At least 0.5 seconds
                                # Transcribe
                                text = self.transcribe(audio)

                                if text:
                                    print_user(text)

                                    # Get Claude's response
                                    response = self.chat(text)
                                    print_claude(response)

                                    # Speak the response
                                    self.speak(response)
                                else:
                                    print_status("No speech detected.", Colors.YELLOW)
                            else:
                                print_status("Recording too short.", Colors.YELLOW)

                            print_status("\nHold SPACE to talk...", Colors.CYAN)

            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        except ImportError:
            print_status("Running in simple mode (press Enter after speaking)...", Colors.YELLOW)
            self._run_simple_mode()

    def _select_stdin(self):
        """Check if stdin has data available."""
        import select
        return select.select([sys.stdin], [], [], 0.1)[0]

    def _run_simple_mode(self):
        """Fallback mode: press Enter to start/stop recording."""
        print_status("\nSimple mode: Press Enter to start recording, Enter again to stop.\n", Colors.CYAN)

        while True:
            try:
                input("Press Enter to start recording (or 'q' + Enter to quit): ")

                user_input = input().strip()
                if user_input.lower() == 'q':
                    print_status("Goodbye!", Colors.YELLOW)
                    break

                print_status("Recording for 5 seconds...", Colors.GREEN)

                # Record for 5 seconds
                audio = sd.rec(
                    int(5 * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=np.float32
                )
                sd.wait()

                print_status("Processing...", Colors.YELLOW)

                text = self.transcribe(audio)

                if text:
                    print_user(text)
                    response = self.chat(text)
                    print_claude(response)
                    self.speak(response)
                else:
                    print_status("No speech detected.", Colors.YELLOW)

            except KeyboardInterrupt:
                print_status("\nGoodbye!", Colors.YELLOW)
                break


def main():
    # Check for required environment variable
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print_status("Error: ANTHROPIC_API_KEY environment variable not set.", Colors.RED)
        print_status("Set it with: export ANTHROPIC_API_KEY=your-key-here", Colors.YELLOW)
        sys.exit(1)

    chat = VoiceChat()
    chat.run()


if __name__ == "__main__":
    main()
