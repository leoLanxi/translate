# Japanese Video Subtitle Generator 🎬

A local Japanese video subtitle generation tool based on faster-whisper, supporting multi-language subtitles, GPU acceleration, real-time progress display, and resume from breakpoint.

**Note: This is a no-burn version. Subtitle burning is disabled by default. Only SRT subtitle files will be generated.**

## ✨ Features

- 🎯 **Japanese Speech Recognition**: Using OpenAI Whisper model, optimized for Japanese
- 🌐 **Multi-language Subtitles**: Support generating Japanese, English, Chinese subtitles (optional)
- 🚀 **GPU Acceleration**: Auto-detect NVIDIA GPU, 5-10x speedup on Windows
- 📊 **Real-time Progress**: Visual progress bar showing processing progress and estimated remaining time
- 💾 **Resume from Breakpoint**: Automatically save progress, continue processing after interruption
- 📁 **Batch Processing**: Automatically process all video files in folder
- 📝 **Standard SRT Output**: Generate universal SRT subtitle files
- ⚠️ **No Subtitle Burning**: This version does not burn subtitles into video (SRT files only)

## 🖥️ Cross-platform Support

| Platform | Computing Device | Relative Speed |
|----------|------------------|----------------|
| Windows + NVIDIA GPU | CUDA (float16) | ⚡ **Fastest (5-10x)** |
| Linux + NVIDIA GPU | CUDA (float16) | ⚡ **Fastest (5-10x)** |
| macOS Apple Silicon | CPU (int8) | 🔹 Baseline speed |
| macOS Intel / Others | CPU (int8) | 🔹 Baseline speed |

The program **automatically detects** your hardware environment and selects the best configuration, no manual setup required.

## 📋 System Requirements

- **Python**: 3.9 or higher
- **ffmpeg**: Not required for this no-burn version (only needed if you want to enable subtitle burning)
- **Disk Space**: Model files ~1-3GB (depends on selected model size)
- **VRAM (GPU users)**: Recommended 8GB+, 16GB can run large-v3 model

## 🚀 Quick Start

### 1. Clone Repository

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

# Windows CUDA acceleration (optional, for GPU users):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

### 4. Prepare Video Files

Put video files to be processed into the `input_videos` folder.

### 5. Run Program

```bash
python main.py
```

## ⚙️ Configuration Options

You can modify the following configuration at the top of `main.py`:

```python
# Input/Output folders
INPUT_DIR = "input_videos"    # Folder containing original videos
OUTPUT_DIR = "output"         # Folder for output subtitles and videos

# Subtitle burning (disabled in this version)
BURN_SUBTITLE = False         # True: Generate video with subtitles, False: Only generate .srt files

# Subtitle language settings (Important!)
SUBTITLE_LANGUAGE = "zh"      # Options: "ja" (Japanese), "en" (English), "zh" (Chinese)

# Model size (affects speed and accuracy)
MODEL_SIZE = "medium"         # Options: tiny, base, small, medium, large-v2, large-v3
```

### Subtitle Language Explanation

| Language Code | Output Subtitle | Implementation |
|---------------|-----------------|----------------|
| `ja` | Japanese (original) | Whisper speech recognition |
| `en` | English | Whisper built-in translation (fast) |
| `zh` | Chinese | Speech recognition + Google translation |

### Model Size Comparison

| Model | Size | VRAM Required | Relative Speed | Japanese Quality |
|-------|------|--------------|----------------|------------------|
| tiny | ~75MB | ~1GB | Fastest | Fair |
| base | ~150MB | ~1GB | Very fast | Good |
| small | ~500MB | ~2GB | Fast | Good |
| medium | ~1.5GB | ~5GB | Medium | Very good |
| large-v3 | ~3GB | ~10GB | Slow | Best |

**Recommendations**:
- Quick preview/test: Use `small`
- Daily use: Use `medium` (balance speed and quality)
- Highest quality: Use `large-v3`

## 📊 Progress Display

The program displays real-time progress when running:

```
  🎤 Starting speech recognition...
  📊 Video duration: 43min 22sec
  Recognition progress: |████████████████████████████████| 2622/2622sec [03:45<00:00, 11.6sec/s]
  ✓ Speech recognition completed, 115 subtitle segments

  📝 Starting subtitle translation (Japanese → Chinese)...
  Translation progress: |████████████████████████████████| 115/115 [00:23<00:00]
  ✓ Translation completed

  ⏱ Total time: 8min 20sec
```

## 💾 Resume from Breakpoint

If processing is interrupted (Ctrl+C or program crash):

1. Progress is automatically saved to `output/progress.json`
2. Next run will automatically detect and continue from breakpoint
3. After selecting continue, processing resumes from last position

```
💾 Found incomplete task: video.mp4
   Progress: 1234.5/2622.0sec
   Will automatically resume...
```

## 📂 Output Files

After processing, the `output` folder will contain:

```
output/
├── video1_zh.srt           # Chinese subtitle file
├── video2_en.srt           # English subtitle file
└── ...
```

**Note**: This version only generates SRT files. Video files with burned subtitles are not generated.

## 🔧 Common Issues

### Q: GPU not detected?

Ensure NVIDIA driver and CUDA are installed:
```bash
# Check if GPU is available
nvidia-smi
```

### Q: Model download failed?

First run will download model from Hugging Face. If network is unstable:
1. Try using proxy or VPN
2. Or manually download model to `~/.cache/huggingface/` directory
3. For users in mainland China, the program automatically uses hf-mirror.com mirror

### Q: Recognition too slow?

1. Use smaller model (like `small` or `base`)
2. If you have NVIDIA GPU, ensure CUDA is working properly
3. faster-whisper is already 4-5x faster than original whisper

### Q: Translation inaccurate?

Translation uses Google Translate free API, may not be accurate for technical terms. You can:
1. Manually edit SRT file after generation
2. Use subtitle editing software (like Aegisub) for fine-tuning

### Q: Want to enable subtitle burning?

1. Install ffmpeg:
   - Windows: `choco install ffmpeg` or download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
2. Set `BURN_SUBTITLE = True` in `main.py`
3. Run program again

## 📜 SRT Subtitle Format

Generated SRT file format:

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
