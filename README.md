# 日语视频字幕自动生成工具 🎬

一个基于 faster-whisper 的本地日语视频字幕生成工具，支持多语言字幕、GPU 加速、实时进度显示和断点续传。

## ✨ 功能特点

- 🎯 **日语语音识别**：使用 OpenAI Whisper 模型，专门针对日语优化
- 🌐 **多语言字幕**：支持生成日语、英语、中文字幕（可选）
- 🚀 **GPU 加速**：自动检测 NVIDIA GPU，Windows 下速度提升 5-10 倍
- 📊 **实时进度**：可视化进度条，清晰显示处理进度和预估剩余时间
- 💾 **断点续传**：处理中断后自动保存进度，下次运行继续处理
- 📁 **批量处理**：自动处理文件夹中所有视频文件
- 📝 **标准 SRT 输出**：生成通用的 SRT 字幕文件
- 🔥 **字幕烧录**：可选将字幕永久嵌入视频（硬字幕）

## 🖥️ 跨平台支持

| 平台 | 计算设备 | 相对速度 |
|------|----------|----------|
| Windows + NVIDIA GPU | CUDA (float16) | ⚡ **最快 (5-10x)** |
| Linux + NVIDIA GPU | CUDA (float16) | ⚡ **最快 (5-10x)** |
| macOS Apple Silicon | CPU (int8) | 🔹 基准速度 |
| macOS Intel / 其他 | CPU (int8) | 🔹 基准速度 |

程序会**自动检测**你的硬件环境并选择最佳配置，无需手动设置。

## 📋 系统要求

- **Python**：3.9 或更高版本
- **ffmpeg**：用于字幕烧录功能
- **磁盘空间**：模型文件约 1-3GB（取决于选择的模型大小）
- **显存（GPU 用户）**：建议 8GB 以上，16GB 可运行 large-v3 模型

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/leoLanxi/translate.git
cd translate
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 ffmpeg

```bash
# macOS
brew install ffmpeg

# Windows (使用 Chocolatey)
choco install ffmpeg

# Windows (手动安装)
# 从 https://ffmpeg.org/download.html 下载并添加到 PATH

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 5. 准备视频文件

将需要处理的视频文件放入 `input_videos` 文件夹。

### 6. 运行程序

```bash
python main.py
```

## ⚙️ 配置选项

在 `main.py` 文件顶部可以修改以下配置：

```python
# 输入/输出文件夹
INPUT_DIR = "input_videos"    # 存放原始视频的文件夹
OUTPUT_DIR = "output"         # 输出字幕和视频的文件夹

# 是否烧录字幕
BURN_SUBTITLE = True          # True: 生成带字幕的视频, False: 仅生成 .srt 文件

# 字幕语言设置（重要！）
SUBTITLE_LANGUAGE = "zh"      # 可选: "ja" (日语), "en" (英语), "zh" (中文)

# 模型大小（影响速度和准确度）
MODEL_SIZE = "medium"         # 可选: tiny, base, small, medium, large-v2, large-v3
```

### 字幕语言说明

| 语言代码 | 输出字幕 | 实现方式 |
|----------|----------|----------|
| `ja` | 日语（原文） | Whisper 语音识别 |
| `en` | 英语 | Whisper 内置翻译（速度快） |
| `zh` | 中文 | 语音识别 + Google 翻译 |

### 模型大小对比

| 模型 | 大小 | 显存需求 | 相对速度 | 日语效果 |
|------|------|----------|----------|----------|
| tiny | ~75MB | ~1GB | 最快 | 一般 |
| base | ~150MB | ~1GB | 很快 | 较好 |
| small | ~500MB | ~2GB | 快 | 好 |
| medium | ~1.5GB | ~5GB | 中等 | 很好 |
| large-v3 | ~3GB | ~10GB | 较慢 | 最好 |

**推荐**：
- 快速预览/测试：使用 `small`
- 日常使用：使用 `medium`（平衡速度和质量）
- 追求最高质量：使用 `large-v3`

## 📊 进度显示

程序运行时会显示实时进度：

```
  🎤 开始语音识别...
  📊 视频时长: 43分22秒
  识别进度: |████████████████████████████████| 2622/2622秒 [03:45<00:00, 11.6秒/s]
  ✓ 语音识别完成，共 115 个字幕片段

  📝 开始翻译字幕 (日语 → 中文)...
  翻译进度: |████████████████████████████████| 115/115 [00:23<00:00]
  ✓ 翻译完成

  🔥 开始烧录字幕到视频...
  烧录进度: |████████████████████████████████| 2622/2622秒 [04:12<00:00]
  ✓ 带字幕视频已保存

  ⏱ 总耗时: 8分20秒
```

## 💾 断点续传

如果处理过程中中断（Ctrl+C 或程序崩溃）：

1. 进度会自动保存到 `output/progress.json`
2. 下次运行程序会自动检测并询问是否继续
3. 选择继续后，会从上次中断的位置继续处理

```
💾 发现未完成的任务: video.mp4
   进度: 1234.5/2622.0秒
   将自动恢复...
```

## 📂 输出文件

处理完成后，`output` 文件夹中会包含：

```
output/
├── video1_zh.srt           # 中文字幕文件
├── video1_zh_subbed.mp4    # 带中文字幕的视频
├── video2_en.srt           # 英文字幕文件
├── video2_en_subbed.mp4    # 带英文字幕的视频
└── ...
```

## 🔧 常见问题

### Q: GPU 没有被检测到？

确保已安装 NVIDIA 驱动和 CUDA：
```bash
# 检查 GPU 是否可用
nvidia-smi
```

### Q: 模型下载失败怎么办？

首次运行时会从 Hugging Face 下载模型，如果网络不稳定：
1. 尝试使用代理或 VPN
2. 或者手动下载模型放到 `~/.cache/huggingface/` 目录

### Q: 识别速度太慢？

1. 使用更小的模型（如 `small` 或 `base`）
2. 如果有 NVIDIA GPU，确保 CUDA 正常工作
3. faster-whisper 已经比原版 whisper 快 4-5 倍

### Q: 翻译不准确？

翻译使用的是 Google 翻译免费接口，对于专业术语可能不够准确。可以：
1. 生成 SRT 文件后手动修改
2. 使用字幕编辑软件（如 Aegisub）微调

### Q: Windows 上 ffmpeg 报错？

确保 ffmpeg 已添加到系统 PATH：
1. 下载 ffmpeg：https://ffmpeg.org/download.html
2. 解压后将 `bin` 目录添加到系统环境变量 PATH

## 📜 SRT 字幕格式

生成的 SRT 文件格式如下：

```srt
1
00:00:01,000 --> 00:00:04,500
这是第一句字幕。

2
00:00:05,200 --> 00:00:08,300
这是第二句字幕。
```

这种格式可以被大多数视频播放器识别（VLC、PotPlayer、IINA 等）。

## 📄 许可证

MIT License - 自由使用和修改

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 强大的语音识别模型
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Whisper 的高性能实现
- [deep-translator](https://github.com/nidhaloff/deep-translator) - 多引擎翻译库
- [FFmpeg](https://ffmpeg.org/) - 多媒体处理工具
