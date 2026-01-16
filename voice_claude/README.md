# Voice Chat with Claude

A simple voice interface for conversing with Claude using local speech-to-text (Whisper) and text-to-speech (Piper).

## Requirements

- Ubuntu 22.04
- NVIDIA GPU (uses GPU 0 for Whisper)
- ANTHROPIC_API_KEY environment variable set

## Setup

### 1. Install system dependencies

```bash
sudo apt install sox libsox-fmt-all portaudio19-dev alsa-utils
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Piper voice model

```bash
mkdir -p ~/.local/share/piper
cd ~/.local/share/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
python voice_chat.py
```

### Controls

- **Hold SPACE** - Record your message
- **Release SPACE** - Send to Claude
- **Press 'q'** - Quit
- **Press 'c'** - Clear conversation history

## Configuration

Edit the top of `voice_chat.py` to customize:

- `WHISPER_MODEL` - Whisper model size (tiny, base, small, medium, large-v2, large-v3)
- `CLAUDE_MODEL` - Claude model to use
- `PIPER_VOICE` - Path to Piper voice model

## Alternative Piper Voices

Browse available voices at: https://huggingface.co/rhasspy/piper-voices/tree/main

Some good options:
- `en_US/lessac/medium` - Natural US English (default)
- `en_US/amy/medium` - Female US English
- `en_GB/alan/medium` - British English
