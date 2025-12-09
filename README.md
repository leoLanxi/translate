# Japanese Video Subtitle Generator 🎬

A local Japanese video subtitle generation tool based on faster-whisper, supporting multi-language subtitles, GPU acceleration, real-time progress display, and resume from breakpoints.

## ✨ Features

- 🎯 **Japanese Speech Recognition**: Uses OpenAI Whisper model, optimized for Japanese
- 🌐 **Multi-language Subtitles**: Generate Japanese, English, or Chinese subtitles
- 🚀 **GPU Acceleration**: Auto-detects NVIDIA GPU, 5-10x faster on Windows with CUDA
- 📊 **Real-time Progress**: Visual progress bar showing processing status and ETA
- 💾 **Resume Support**: Auto-saves progress, continues from where you left off
- 📁 **Batch Processing**: Automatically processes all video files in a folder
- 📝 **Standard SRT Output**: Generates universal SRT subtitle files
- 🔥 **Subtitle Burning**: Optionally burn subtitles permanently into video (hardcoded)

## 🖥️ Cross-Platform Support

| Platform | Compute Device | Relative Speed |
|----------|----------------|----------------|
| Windows + NVIDIA GPU | CUDA (float16) | ⚡ **Fastest (5-10x)** |
| Linux + NVIDIA GPU | CUDA (float16) | ⚡ **Fastest (5-10x)** |
| macOS Apple Silicon | CPU (int8) | 🔹 Baseline |
| macOS Intel / Others | CPU (int8) | 🔹 Baseline |

The program **automatically detects** your hardware and selects the best configuration.

## 📋 Requirements

- **Python**: 3.9 or higher
- **ffmpeg**: For subtitle burning feature
- **Disk Space**: ~1-3GB for model files (depends on model size)
- **VRAM (GPU users)**: 8GB+ recommended, 16GB can run large-v3 model

## 🚀 Quick Start

### 1. Clone the Project

```bash
git clone https://github.com/leoLanxi/translate.git
cd translate
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg

# Windows (manual)
# Download from https://ffmpeg.org/download.html and add to PATH

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 5. Prepare Video Files

Place your video files in the `input_videos` folder.

### 6. Run

```bash
python main.py
```

## ⚙️ Configuration

Edit the configuration section at the top of `main.py`:

```python
# Input/Output folders
INPUT_DIR = "input_videos"    # Folder containing source videos
OUTPUT_DIR = "output"         # Output folder for subtitles and videos

# Burn subtitles into video
BURN_SUBTITLE = True          # True: generate video with hardcoded subs, False: SRT only

# Subtitle language (Important!)
SUBTITLE_LANGUAGE = "zh"      # Options: "ja" (Japanese), "en" (English), "zh" (Chinese)

# Model size (affects speed and accuracy)
MODEL_SIZE = "medium"         # Options: tiny, base, small, medium, large-v2, large-v3
```

### Subtitle Language Options

| Code | Output | Implementation |
|------|--------|----------------|
| `ja` | Japanese (original) | Whisper transcription |
| `en` | English | Whisper built-in translation (fast) |
| `zh` | Chinese | Transcription + Google Translate |

### Model Size Comparison

| Model | Size | VRAM | Speed | Japanese Quality |
|-------|------|------|-------|------------------|
| tiny | ~75MB | ~1GB | Fastest | Fair |
| base | ~150MB | ~1GB | Very Fast | Good |
| small | ~500MB | ~2GB | Fast | Better |
| medium | ~1.5GB | ~5GB | Medium | Very Good |
| large-v3 | ~3GB | ~10GB | Slower | Best |

**Recommendations**:
- Quick preview/testing: `small`
- Daily use: `medium` (balanced speed and quality)
- Best quality: `large-v3`

## 📊 Progress Display

The program shows real-time progress:

```
  🎤 Starting speech recognition...
  📊 Video duration: 43m 22s
  Recognition: |████████████████████████| 2622/2622s [03:45<00:00, 11.6s/s]
  ✓ Recognition complete, 115 subtitle segments

  📝 Translating subtitles (Japanese → Chinese)...
  Translation: |████████████████████████| 115/115 [00:23<00:00]
  ✓ Translation complete

  🔥 Burning subtitles to video...
  Burning: |████████████████████████| 2622/2622s [04:12<00:00]
  ✓ Video with subtitles saved

  ⏱ Total time: 8m 20s
```

## 💾 Resume from Breakpoint

If processing is interrupted (Ctrl+C or crash):

1. Progress is automatically saved to `output/progress.json`
2. Next run will detect and offer to resume
3. Processing continues from where it left off

```
💾 Found incomplete task: video.mp4
   Progress: 1234.5/2622.0s
   Resuming automatically...
```

## 📂 Output Files

After processing, the `output` folder contains:

```
output/
├── video1_zh.srt           # Chinese subtitle file
├── video1_zh_subbed.mp4    # Video with Chinese subtitles
├── video2_en.srt           # English subtitle file
├── video2_en_subbed.mp4    # Video with English subtitles
└── ...
```

## 🔧 Troubleshooting

### Q: GPU not detected?

Make sure NVIDIA drivers and CUDA are installed:
```bash
# Check if GPU is available
nvidia-smi
```

### Q: Model download failed?

The model downloads from Hugging Face on first run. If network is unstable:
1. Try using a proxy or VPN
2. Or manually download the model to `~/.cache/huggingface/`

### Q: Processing too slow?

1. Use a smaller model (`small` or `base`)
2. If you have NVIDIA GPU, ensure CUDA is working
3. faster-whisper is already 4-5x faster than original whisper

### Q: Translation not accurate?

Translation uses Google Translate free API, which may not be accurate for technical terms. You can:
1. Edit the SRT file manually after generation
2. Use subtitle editing software (e.g., Aegisub) for fine-tuning

### Q: ffmpeg error on Windows?

Make sure ffmpeg is added to system PATH:
1. Download ffmpeg: https://ffmpeg.org/download.html
2. Extract and add the `bin` directory to system PATH

## 📜 SRT Format

Generated SRT files follow this format:

```srt
1
00:00:01,000 --> 00:00:04,500
This is the first subtitle.

2
00:00:05,200 --> 00:00:08,300
This is the second subtitle.
```

This format is recognized by most video players (VLC, PotPlayer, IINA, etc.).

## 📄 License

MIT License - Free to use and modify

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Powerful speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - High-performance Whisper implementation
- [deep-translator](https://github.com/nidhaloff/deep-translator) - Multi-engine translation library
- [FFmpeg](https://ffmpeg.org/) - Multimedia processing tool
