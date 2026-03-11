
# BaudShift: Information-Dense Audio Optimizer (MVP)

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

BaudShift is an audio processing tool that automatically adjusts the playback speed of spoken-word audio based on its cognitive density. 
Inspired by the idea that a universally optimal information density for spoken language may exist, https://www.science.org/doi/10.1126/sciadv.aaw2594, BaudShift approximates audio input complexity and adjusts the playback speed accordingly.
Instead of blindly cutting silence, this script uses local AI models to evaluate the "perplexity" (predictability) of the transcribed speech. It speeds up conversational "fluff" and plays dense, technical information at a normal pace, helping listeners absorb information at an optimal cognitive rate (aiming toward the universal ~39 bits per second).

## 🚀 How It Works

1. **Transcription:** Uses `faster-whisper` (Tiny model) to generate a fast, local transcript of the input audio.
2. **Density Scoring:** Passes the transcribed segments through a local `GPT-2` model to calculate the **Perplexity** of the text. 
   * *Low Perplexity (< 25):* Predictable, conversational, easy to process.
   * *High Perplexity (> 60):* Dense, technical, unpredictable phrasing.
3. **Decision Engine:** Calculates the median perplexity of the entire file to assign a global speed multiplier (ranging from 1.0x to 1.75x).
4. **Audio Processing:** Uses `ffmpeg` to seamlessly adjust the tempo of the entire audio file without distorting the pitch, outputting a highly optimized MP3.

## 🛠️ Prerequisites

You will need Python 3.8+ and the following system dependencies:

### 1. FFmpeg (Required for Audio Processing)
The script relies on `ffmpeg` to handle the final audio tempo adjustment. 
* **Windows:** You can easily install this via Winget by opening your terminal and running: `winget install ffmpeg`. Ensure it is added to your system PATH.
* **macOS:** `brew install ffmpeg`
* **Linux:** `sudo apt install ffmpeg`

### 2. Python Packages
Install the required Python libraries using pip:
```bash
pip install torch numpy faster-whisper transformers
```

## 💻 Usage

1. Place your target audio file in the same directory as the script.
2. Open the script and modify the configuration block at the top to match your files:
```python
# --- CONFIGURATION ---
AUDIO_FILE = "your_input_audio.m4a"
OUTPUT_FILE = "optimized_output.mp3"
# ---------------------
```


3. Run the script:
```bash
python baudshift.py
```


4. The terminal will output a **Report Card** displaying the median density score, the resulting verdict, and the applied speed multiplier before saving the final file.

## 🎛️ Tuning the Engine

The thresholds for determining what constitutes "fluff" versus "dense" material are located in the `recommend_speed` function. You can tweak these values based on your personal cognitive preferences or the specific domain of your audio (e.g., highly technical lectures vs. casual podcasts).

```python
if avg_perplexity < 25:
    return 1.75, "Very Easy (Fluff)"
elif avg_perplexity < 45:
    return 1.5, "Easy (Conversational)"
```
