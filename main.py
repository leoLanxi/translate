#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日语视频字幕自动生成工具 v3.0（多语言 + GPU加速 + 进度显示）
============================================================

功能说明：
    1. 批量读取指定文件夹中的视频文件（支持 mp4, mkv, avi, mov）
    2. 使用 faster-whisper 模型进行日语语音识别
    3. 支持生成日语/英语/中文字幕（可选）
    4. 自动检测 GPU/CPU，Windows CUDA 自动加速
    5. 实时进度条显示处理进度
    6. 支持断点续传（中断后可继续）
    7. 可选将字幕烧录（硬字幕）到视频中

跨平台支持：
    - macOS (Apple Silicon M1/M2/M3) - 使用 CPU
    - Windows (NVIDIA GPU) - 使用 CUDA 加速
    - Linux (NVIDIA GPU) - 使用 CUDA 加速

安装依赖：
    pip install -r requirements.txt

    # Windows CUDA 加速额外安装（可选）：
    pip install nvidia-cublas-cu11 nvidia-cudnn-cu11

运行方式：
    python main.py

作者：AI Assistant
日期：2024
"""

import os
import sys
import json
import subprocess
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

# ==================== 配置区域 ====================
# 修改下面的变量来自定义程序行为

# 输入文件夹：存放需要处理的视频文件
INPUT_DIR = "input_videos"

# 输出文件夹：存放生成的 .srt 字幕文件和（可选的）烧录字幕后的视频
OUTPUT_DIR = "output"

# 是否将字幕烧录到视频中（生成带硬字幕的新视频）
BURN_SUBTITLE = True

# 字幕语言设置
# 可选值：
#   - "ja": 日语字幕（原文，仅语音识别）
#   - "en": 英语字幕（使用 Whisper 内置翻译）
#   - "zh": 中文字幕（语音识别后翻译成中文）
SUBTITLE_LANGUAGE = "zh"

# Whisper 模型大小
# 可选值："tiny", "base", "small", "medium", "large-v2", "large-v3"
# 模型越大，识别效果越好，但速度越慢，占用内存越多
# 推荐：
#   - 快速处理：使用 "small" 或 "base"
#   - 高质量：使用 "medium" 或 "large-v3"
MODEL_SIZE = "medium"

# 支持的视频格式
SUPPORTED_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv")

# 字幕字体设置（用于烧录字幕时）
SUBTITLE_FONT_SIZE = 24
SUBTITLE_FONT_COLOR = "white"
SUBTITLE_OUTLINE_COLOR = "black"
SUBTITLE_OUTLINE_WIDTH = 2

# 断点续传：保存进度的文件
PROGRESS_FILE = "progress.json"

# ==================== 配置区域结束 ====================


@dataclass
class Segment:
    """字幕片段数据类"""
    start: float
    end: float
    text: str


@dataclass
class ProcessingProgress:
    """处理进度数据类（用于断点续传）"""
    video_path: str
    segments: List[dict]  # 已识别的字幕片段
    last_position: float  # 最后处理到的位置（秒）
    total_duration: float  # 视频总时长
    is_transcribed: bool  # 是否完成语音识别
    is_translated: bool  # 是否完成翻译
    subtitle_lang: str  # 字幕语言


def detect_device() -> Tuple[str, str]:
    """
    自动检测最佳计算设备
    
    Returns:
        Tuple[str, str]: (device, compute_type)
        - CUDA GPU: ("cuda", "float16")
        - CPU: ("cpu", "int8")
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ 检测到 NVIDIA GPU: {gpu_name}")
            print("  将使用 CUDA 加速（速度提升 5-10 倍）")
            return "cuda", "float16"
    except ImportError:
        pass
    
    # 检查是否有 CUDA 可用（不依赖 torch）
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            print(f"✓ 检测到 NVIDIA GPU: {gpu_name}")
            print("  将使用 CUDA 加速（速度提升 5-10 倍）")
            return "cuda", "float16"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # 没有 GPU，使用 CPU
    import platform
    if platform.processor() == 'arm' or 'Apple' in platform.processor():
        print("✓ 检测到 Apple Silicon，使用 CPU 优化模式")
    else:
        print("✓ 未检测到 NVIDIA GPU，使用 CPU 模式")
    return "cpu", "int8"


def check_dependencies() -> bool:
    """检查必要的依赖是否已安装"""
    print("=" * 60)
    print("检查依赖...")
    print("=" * 60)
    
    all_ok = True
    
    # 检查 faster-whisper
    try:
        from faster_whisper import WhisperModel
        print("✓ faster-whisper 已安装")
    except ImportError:
        print("✗ faster-whisper 未安装")
        print("  请运行: pip install faster-whisper")
        all_ok = False
    
    # 检查 tqdm（进度条）
    try:
        from tqdm import tqdm
        print("✓ tqdm 已安装（进度条）")
    except ImportError:
        print("✗ tqdm 未安装（进度条显示需要）")
        print("  请运行: pip install tqdm")
        all_ok = False
    
    # 检查翻译库（仅在需要中文字幕时）
    if SUBTITLE_LANGUAGE == "zh":
        try:
            from deep_translator import GoogleTranslator
            print("✓ deep-translator 已安装（中文翻译）")
        except ImportError:
            print("✗ deep-translator 未安装（中文字幕需要）")
            print("  请运行: pip install deep-translator")
            all_ok = False
    
    # 检查 ffmpeg（仅在需要烧录字幕时）
    if BURN_SUBTITLE:
        if shutil.which("ffmpeg"):
            print("✓ ffmpeg 已安装")
        else:
            print("✗ ffmpeg 未安装（烧录字幕功能需要）")
            if sys.platform == "darwin":
                print("  请运行: brew install ffmpeg")
            elif sys.platform == "win32":
                print("  请从 https://ffmpeg.org/download.html 下载并添加到 PATH")
            else:
                print("  请运行: sudo apt install ffmpeg")
            all_ok = False
    
    print()
    return all_ok


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_duration(seconds: float) -> str:
    """将秒数格式化为易读的时长字符串"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}分{seconds % 60:.0f}秒"
    else:
        return f"{seconds // 3600:.0f}时{(seconds % 3600) // 60:.0f}分"


def save_progress(progress: ProcessingProgress, output_dir: str) -> None:
    """保存处理进度（用于断点续传）"""
    progress_path = os.path.join(output_dir, PROGRESS_FILE)
    data = {
        "video_path": progress.video_path,
        "segments": progress.segments,
        "last_position": progress.last_position,
        "total_duration": progress.total_duration,
        "is_transcribed": progress.is_transcribed,
        "is_translated": progress.is_translated,
        "subtitle_lang": progress.subtitle_lang,
        "saved_at": datetime.now().isoformat()
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_progress(output_dir: str) -> Optional[ProcessingProgress]:
    """加载之前保存的进度"""
    progress_path = os.path.join(output_dir, PROGRESS_FILE)
    if not os.path.exists(progress_path):
        return None
    
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return ProcessingProgress(
            video_path=data["video_path"],
            segments=data["segments"],
            last_position=data["last_position"],
            total_duration=data["total_duration"],
            is_transcribed=data["is_transcribed"],
            is_translated=data["is_translated"],
            subtitle_lang=data["subtitle_lang"]
        )
    except Exception as e:
        print(f"⚠ 无法加载进度文件: {e}")
        return None


def clear_progress(output_dir: str) -> None:
    """清除进度文件"""
    progress_path = os.path.join(output_dir, PROGRESS_FILE)
    if os.path.exists(progress_path):
        os.remove(progress_path)


def load_whisper_model(model_size: str = MODEL_SIZE):
    """加载 Whisper 语音识别模型（自动选择最佳设备）"""
    from faster_whisper import WhisperModel
    
    print(f"\n正在加载 Whisper 模型 ({model_size})...")
    print("提示：首次运行时会自动下载模型，请确保网络连接正常")
    
    # 自动检测设备
    device, compute_type = detect_device()
    
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    
    print(f"✓ 模型加载完成 (设备: {device}, 精度: {compute_type})")
    return model


def translate_text(text: str, target_lang: str = "zh") -> str:
    """将日语文本翻译成目标语言"""
    if not text.strip():
        return text
    
    try:
        from deep_translator import GoogleTranslator
        
        lang_map = {"zh": "zh-CN", "en": "en"}
        target = lang_map.get(target_lang, "zh-CN")
        
        translator = GoogleTranslator(source='ja', target=target)
        translated = translator.translate(text)
        return translated if translated else text
    except Exception as e:
        return text  # 翻译失败时返回原文


def translate_segments_with_progress(
    segments: List[Segment], 
    target_lang: str = "zh",
    progress_callback=None
) -> List[Segment]:
    """批量翻译字幕片段（带进度显示）"""
    from tqdm import tqdm
    
    print(f"\n  📝 开始翻译字幕 (日语 → {'中文' if target_lang == 'zh' else '英语'})...")
    
    translated_segments = []
    
    with tqdm(total=len(segments), desc="  翻译进度", unit="条", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for seg in segments:
            translated_text = translate_text(seg.text, target_lang)
            translated_segments.append(Segment(
                start=seg.start,
                end=seg.end,
                text=translated_text
            ))
            pbar.update(1)
            
            if progress_callback:
                progress_callback(len(translated_segments), len(segments))
    
    print(f"  ✓ 翻译完成")
    return translated_segments


def transcribe_video_with_progress(
    video_path: str,
    model,
    language: str = "ja",
    subtitle_lang: str = "ja",
    output_dir: str = None
) -> Tuple[List[Segment], float]:
    """
    对视频进行语音识别（带实时进度显示）
    
    Returns:
        Tuple[List[Segment], float]: (字幕片段列表, 视频总时长)
    """
    from tqdm import tqdm
    
    # 确定任务类型
    if subtitle_lang == "en":
        task = "translate"
        print(f"\n  🎤 开始语音识别并翻译成英语...")
    else:
        task = "transcribe"
        print(f"\n  🎤 开始语音识别...")
    
    # 获取视频时长
    segments_generator, info = model.transcribe(
        video_path,
        language=language,
        task=task,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    total_duration = info.duration
    print(f"  📊 视频时长: {format_duration(total_duration)}")
    
    segments = []
    last_end = 0
    
    # 创建进度条
    with tqdm(total=int(total_duration), desc="  识别进度", unit="秒",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}秒 [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for segment in segments_generator:
            seg = Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip()
            )
            segments.append(seg)
            
            # 更新进度条
            progress = int(segment.end) - int(last_end)
            if progress > 0:
                pbar.update(progress)
                last_end = segment.end
            
            # 定期保存进度（每30秒保存一次）
            if output_dir and len(segments) % 10 == 0:
                progress_data = ProcessingProgress(
                    video_path=video_path,
                    segments=[asdict(s) for s in segments],
                    last_position=segment.end,
                    total_duration=total_duration,
                    is_transcribed=False,
                    is_translated=False,
                    subtitle_lang=subtitle_lang
                )
                save_progress(progress_data, output_dir)
        
        # 确保进度条到100%
        remaining = int(total_duration) - int(last_end)
        if remaining > 0:
            pbar.update(remaining)
    
    print(f"  ✓ 语音识别完成，共 {len(segments)} 个字幕片段")
    return segments, total_duration


def write_srt(segments: List[Segment], srt_path: str) -> None:
    """将字幕片段列表写入 SRT 文件"""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            f.write(f"{i}\n")
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment.text}\n")
            f.write("\n")
    
    print(f"  ✓ 字幕文件已保存: {srt_path}")


def burn_subtitles_with_progress(
    video_path: str,
    srt_path: str,
    output_video_path: str,
    font_size: int = SUBTITLE_FONT_SIZE,
) -> bool:
    """使用 ffmpeg 将字幕烧录到视频中（带进度显示）"""
    from tqdm import tqdm
    
    print(f"\n  🔥 开始烧录字幕到视频...")
    
    if not shutil.which("ffmpeg"):
        print("  ✗ 错误：ffmpeg 未安装")
        return False
    
    # 获取视频时长
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        total_duration = float(result.stdout.strip())
    except:
        total_duration = 0
    
    # 处理路径（Windows 兼容）
    if sys.platform == "win32":
        # Windows 下需要特殊处理路径
        srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    else:
        srt_path_escaped = srt_path.replace("'", r"\'").replace(":", r"\:")
    
    subtitle_filter = (
        f"subtitles='{srt_path_escaped}':"
        f"force_style='FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"Outline={SUBTITLE_OUTLINE_WIDTH},"
        f"BorderStyle=1'"
    )
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitle_filter,
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-y",
        "-progress", "pipe:1",  # 输出进度信息
        output_video_path
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        with tqdm(total=int(total_duration) if total_duration else 100, 
                  desc="  烧录进度", unit="秒",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}秒 [{elapsed}<{remaining}]') as pbar:
            
            current_time = 0
            for line in process.stdout:
                if line.startswith("out_time_ms="):
                    try:
                        time_ms = int(line.split("=")[1])
                        new_time = time_ms // 1000000
                        if new_time > current_time:
                            pbar.update(new_time - current_time)
                            current_time = new_time
                    except:
                        pass
            
            process.wait()
            
            # 确保进度条完成
            if total_duration and current_time < int(total_duration):
                pbar.update(int(total_duration) - current_time)
        
        if process.returncode == 0:
            print(f"  ✓ 带字幕视频已保存: {output_video_path}")
            return True
        else:
            print(f"  ✗ ffmpeg 执行失败")
            return False
            
    except Exception as e:
        print(f"  ✗ 烧录失败: {str(e)}")
        return False


def process_single_video(
    video_path: str,
    output_dir: str,
    model,
    burn_subtitle: bool = BURN_SUBTITLE,
    subtitle_lang: str = SUBTITLE_LANGUAGE,
    resume_progress: ProcessingProgress = None
) -> bool:
    """处理单个视频文件（支持断点续传）"""
    video_name = Path(video_path).stem
    lang_suffix = {"ja": "_ja", "en": "_en", "zh": "_zh"}.get(subtitle_lang, "")
    srt_path = os.path.join(output_dir, f"{video_name}{lang_suffix}.srt")
    lang_name = {"ja": "日语", "en": "英语", "zh": "中文"}.get(subtitle_lang, subtitle_lang)
    
    print(f"\n{'='*60}")
    print(f"📹 处理视频: {video_path}")
    print(f"🌐 字幕语言: {lang_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 检查是否有可恢复的进度
        if resume_progress and resume_progress.video_path == video_path:
            print(f"\n  📂 发现未完成的进度，从断点继续...")
            segments = [Segment(**s) for s in resume_progress.segments]
            total_duration = resume_progress.total_duration
            
            if not resume_progress.is_transcribed:
                # 需要重新识别（当前不支持部分恢复，重新开始）
                print(f"  ⚠ 语音识别未完成，重新开始...")
                segments, total_duration = transcribe_video_with_progress(
                    video_path, model, 
                    language="ja", 
                    subtitle_lang=subtitle_lang,
                    output_dir=output_dir
                )
        else:
            # 1. 语音识别
            segments, total_duration = transcribe_video_with_progress(
                video_path, model, 
                language="ja", 
                subtitle_lang=subtitle_lang,
                output_dir=output_dir
            )
        
        if not segments:
            print("  ⚠ 警告：未识别到任何语音内容")
            clear_progress(output_dir)
            return True
        
        # 保存识别完成状态
        progress_data = ProcessingProgress(
            video_path=video_path,
            segments=[asdict(s) for s in segments],
            last_position=total_duration,
            total_duration=total_duration,
            is_transcribed=True,
            is_translated=False,
            subtitle_lang=subtitle_lang
        )
        save_progress(progress_data, output_dir)
        
        # 2. 如果需要中文字幕，进行翻译
        if subtitle_lang == "zh":
            if resume_progress and resume_progress.is_translated:
                print(f"  ✓ 翻译已完成（从缓存加载）")
            else:
                segments = translate_segments_with_progress(segments, target_lang="zh")
                # 保存翻译完成状态
                progress_data.segments = [asdict(s) for s in segments]
                progress_data.is_translated = True
                save_progress(progress_data, output_dir)
        
        # 3. 生成 SRT 字幕文件
        write_srt(segments, srt_path)
        
        # 4. 可选：烧录字幕
        if burn_subtitle:
            output_video_path = os.path.join(output_dir, f"{video_name}{lang_suffix}_subbed.mp4")
            burn_subtitles_with_progress(video_path, srt_path, output_video_path)
        
        # 处理完成，清除进度文件
        clear_progress(output_dir)
        
        elapsed = time.time() - start_time
        print(f"\n  ⏱ 总耗时: {format_duration(elapsed)}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n  ⚠ 用户中断！进度已保存，下次运行将自动恢复。")
        raise
    except Exception as e:
        print(f"\n  ✗ 处理失败: {str(e)}")
        print(f"  💡 提示: 进度已保存，修复问题后重新运行即可继续。")
        import traceback
        traceback.print_exc()
        return False


def process_folder(
    input_dir: str,
    output_dir: str,
    burn_subtitle: bool = BURN_SUBTITLE,
    subtitle_lang: str = SUBTITLE_LANGUAGE
) -> None:
    """批量处理文件夹中的所有视频文件"""
    
    # 检查输入文件夹
    if not os.path.exists(input_dir):
        print(f"✗ 错误：输入文件夹不存在: {input_dir}")
        os.makedirs(input_dir, exist_ok=True)
        print(f"  已自动创建空文件夹: {input_dir}")
        return
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 输出文件夹: {output_dir}")
    
    # 查找所有视频文件
    video_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ])
    
    if not video_files:
        print(f"⚠ 警告：在 {input_dir} 中没有找到视频文件")
        print(f"  支持的格式: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"\n📋 找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        print(f"  • {vf}")
    
    # 检查是否有未完成的进度
    saved_progress = load_progress(output_dir)
    if saved_progress:
        print(f"\n💾 发现未完成的任务: {Path(saved_progress.video_path).name}")
        print(f"   进度: {saved_progress.last_position:.1f}/{saved_progress.total_duration:.1f}秒")
        print(f"   将自动恢复...")
    
    # 加载模型
    try:
        model = load_whisper_model(MODEL_SIZE)
    except Exception as e:
        print(f"✗ 模型加载失败: {str(e)}")
        print("  请检查网络连接或尝试使用更小的模型（如 'base'）")
        return
    
    # 处理统计
    success_count = 0
    fail_count = 0
    total_start_time = time.time()
    
    # 逐个处理视频
    try:
        for i, video_file in enumerate(video_files, start=1):
            print(f"\n\n{'#'*60}")
            print(f"# 处理进度: [{i}/{len(video_files)}]")
            print(f"{'#'*60}")
            
            video_path = os.path.join(input_dir, video_file)
            
            # 检查是否是要恢复的视频
            resume = saved_progress if (saved_progress and 
                                         saved_progress.video_path == video_path) else None
            
            if process_single_video(video_path, output_dir, model, 
                                    burn_subtitle, subtitle_lang, resume):
                success_count += 1
            else:
                fail_count += 1
            
            saved_progress = None  # 只对第一个匹配的视频使用恢复
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("⚠ 处理被中断！")
        print(f"{'='*60}")
        print(f"  已完成: {success_count} 个")
        print(f"  未完成: {len(video_files) - success_count - fail_count} 个")
        print(f"  💡 下次运行将自动从断点继续")
        return
    
    # 打印总结
    total_elapsed = time.time() - total_start_time
    lang_name = {"ja": "日语", "en": "英语", "zh": "中文"}.get(subtitle_lang, subtitle_lang)
    
    print(f"\n\n{'='*60}")
    print("🎉 处理完成！")
    print(f"{'='*60}")
    print(f"  ✓ 成功: {success_count} 个")
    print(f"  ✗ 失败: {fail_count} 个")
    print(f"  🌐 字幕语言: {lang_name}")
    print(f"  ⏱ 总耗时: {format_duration(total_elapsed)}")
    print(f"  📂 输出目录: {output_dir}")


def main():
    """程序主入口"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     日语视频字幕自动生成工具 v3.0                          ║
    ║     Japanese Video Subtitle Generator                      ║
    ║                                                            ║
    ║     🚀 支持 GPU 加速 | 📊 实时进度 | 💾 断点续传           ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # 字幕语言名称映射
    lang_names = {"ja": "日语", "en": "英语", "zh": "中文"}
    lang_name = lang_names.get(SUBTITLE_LANGUAGE, SUBTITLE_LANGUAGE)
    
    # 打印当前配置
    print("📋 当前配置:")
    print(f"  • 输入文件夹: {INPUT_DIR}")
    print(f"  • 输出文件夹: {OUTPUT_DIR}")
    print(f"  • 模型大小: {MODEL_SIZE}")
    print(f"  • 字幕语言: {lang_name} ({SUBTITLE_LANGUAGE})")
    print(f"  • 烧录字幕: {'是' if BURN_SUBTITLE else '否'}")
    print()
    
    # 验证字幕语言设置
    if SUBTITLE_LANGUAGE not in ["ja", "en", "zh"]:
        print(f"✗ 错误：不支持的字幕语言 '{SUBTITLE_LANGUAGE}'")
        print("  支持的语言: ja (日语), en (英语), zh (中文)")
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        print("请先安装缺失的依赖，然后重新运行程序。")
        sys.exit(1)
    
    # 开始处理
    process_folder(INPUT_DIR, OUTPUT_DIR, BURN_SUBTITLE, SUBTITLE_LANGUAGE)


if __name__ == "__main__":
    main()
