import torch
import numpy as np
import subprocess
import os
from faster_whisper import WhisperModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --- CONFIGURATION ---
AUDIO_FILE = "high_low_high_info.m4a"
OUTPUT_FILE = "optimized_lecture_highlowhighInfo.mp3"
# ---------------------

def get_perplexity(text, model, tokenizer):
    if not text.strip(): return 0
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    if input_ids.shape[1] == 0: return 0
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()

def recommend_speed(avg_perplexity):
    """
    The Decision Engine.
    Adjust these thresholds based on your personal preference.
    """
    # Thresholds based on GPT-2 typical values
    # < 20: Very conversational / Fluff
    # 20 - 60: Normal speech
    # > 60: Dense technical / unexpected phrasing
    
    if avg_perplexity < 25:
        return 1.75, "Very Easy (Fluff)"
    elif avg_perplexity < 45:
        return 1.5, "Easy (Conversational)"
    elif avg_perplexity < 70:
        return 1.25, "Moderate (Standard)"
    else:
        return 1.0, "Dense (Technical/Complex)"

def process_entire_file(input_file, output_file, speed):
    """
    Applies a single, high-quality speed filter to the whole file.
    No chopping, no stitching, no glitches.
    """
    print(f"--- Rendering final audio at {speed}x ---")
    
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_file,
        "-filter:a", f"atempo={speed}",
        "-c:a", "libmp3lame", "-q:a", "2",
        output_file
    ]
    subprocess.run(cmd, check=True)

def main():
    print(f"--- 1. Scanning Content Density ---")
    whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_model.eval()

    # Transcribe the whole thing to get the text
    segments, _ = whisper.transcribe(AUDIO_FILE, beam_size=5)
    
    all_perplexities = []
    full_text = []

    print("Analyzing segments...")
    for seg in segments:
        text = seg.text.strip()
        if len(text) < 5: continue
        
        ppl = get_perplexity(text, gpt_model, tokenizer)
        all_perplexities.append(ppl)
        full_text.append(text)

    if not all_perplexities:
        print("Error: No speech found.")
        return

    # Calculate Global Stats
    avg_ppl = np.mean(all_perplexities)
    median_ppl = np.median(all_perplexities)
    
    # We use Median to avoid one crazy sentence skewing the score
    target_speed, category = recommend_speed(median_ppl)

    print("\n" + "="*40)
    print(f"REPORT CARD: {AUDIO_FILE}")
    print(f"Median Density Score: {median_ppl:.2f}")
    print(f"Verdict: {category}")
    print(f"Recommended Speed: {target_speed}x")
    print("="*40 + "\n")

    # Generate the file
    process_entire_file(AUDIO_FILE, OUTPUT_FILE, target_speed)
    print(f"Done! Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()