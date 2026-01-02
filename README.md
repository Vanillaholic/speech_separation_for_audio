# README.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个说话人分离（Speaker Diarization）项目，可以从音频/视频文件中分离不同的说话人。提供三种使用接口：
1. **ComfyUI自定义节点** (`nodes.py`) - 集成到ComfyUI工作流中
2. **Gradio网页界面** (`gradio_app.py`) - 独立的Web应用，支持两种输出模式
3. **命令行脚本** (`app.py`) - 基础CLI工具

项目使用 `pyannote.audio` v3.1 进行说话人分离，需要Hugging Face Token并获得 `pyannote/speaker-diarization-3.1` 模型的访问权限。

## 环境配置

### 虚拟环境
项目使用 `venv_portable/` 作为便携式虚拟环境。

**激活虚拟环境:**
```bash
# macOS/Linux
source venv_portable/bin/activate

# Windows
venv_portable\Scripts\activate.bat
```

### 安装依赖
```bash
pip install -r requirements.txt
```

**核心依赖:**
- `torch>=1.9.0,<2.8.0` - PyTorch模型推理
- `pyannote.audio>=3.1.0` - 说话人分离管道
- `soundfile>=0.10.0` - 音频读写
- `gradio==4.19.0` - Web界面
- `moviepy==1.0.3` - 视频音频提取

### Hugging Face Token配置
在项目根目录创建或编辑 `config.json`:
```json
{
    "hf_token": "hf_YOUR_TOKEN_HERE"
}
```

**重要提示:**
- `config.json` 已在 `.gitignore` 中，不会被提交到仓库
- 使用前必须在 https://huggingface.co/pyannote/speaker-diarization-3.1 接受用户协议
- 在 https://huggingface.co/settings/tokens 创建你的token

## 运行应用

### Gradio网页界面（推荐）
```bash
python gradio_app.py
```
应用会自动从7860端口开始查找可用端口并在浏览器中启动。

**功能特性:**
- 支持上传音频或视频文件
- 两种输出模式:
  - 模式1: 与原音频等长，非说话人时段为静音
  - 模式2: 仅拼接说话片段，无静音
- 可调节的说话人检测参数

### 命令行脚本
```bash
python app.py
```
运行前需要在 `app.py` 中配置输入文件路径和HF token。

### ComfyUI节点
1. 将整个文件夹复制到 `ComfyUI/custom_nodes/`
2. 重启ComfyUI
3. 在 `audio/speaker_separation` 分类下找到节点

**可用节点:**
- `SpeakerSeparationNode` - 输出最多5个说话人音轨
- `SpeakerSeparationSingleNode` - 输出单个选定的说话人音轨

## 架构设计

### 核心组件

**SpeakerSeparator类** (`gradio_app.py:34-334`)
- 主要的说话人分离逻辑
- 支持音频和视频输入（使用moviepy从视频提取音频）
- 缓存pyannote管道以避免重复加载模型
- 从 `config.json` 读取HF token

**SpeakerSeparationNode类** (`nodes.py:10-273`)
- ComfyUI节点实现
- 接受AUDIO类型输入，返回5个AUDIO输出（speaker_0到speaker_4）
- 处理ComfyUI的各种音频格式约定（元组、字典、张量）

### 音频处理流程

1. **输入处理:**
   - 音频文件: 使用soundfile直接读取
   - 视频文件: 使用moviepy的VideoFileClip提取音频
   - 立体声转单声道（平均通道）
   - 确保为float32 numpy数组格式

2. **说话人分离:**
   - 将音频保存为临时WAV文件（pyannote需要文件输入）
   - 运行pyannote管道，配置可调参数
   - 管道返回按说话人标记的时间段

3. **输出生成:**
   - **模式1（全长）:** 创建与输入等长的零数组，将说话人片段复制到对应时间位置
   - **模式2（紧凑）:** 仅拼接说话人的片段，无静音填充
   - 每个说话人保存为独立的WAV文件到 `speaker_output/`

### 参数调优

pyannote管道接受以下参数（在 `nodes.py:171-197` 和 `gradio_app.py:202-227` 中配置）:
- `onset` / `offset`: 语音活动检测阈值 (0.0-1.0)
- `min_duration_on` / `min_duration_off`: 最小语音/静音时长（秒）
- `clustering_threshold`: 说话人聚类敏感度
- `num_speakers`: 预期说话人数量（帮助提高模型准确性）

### 文件组织

```
.
├── __init__.py              # ComfyUI节点注册
├── nodes.py                 # ComfyUI节点实现
├── gradio_app.py            # Gradio网页界面
├── app.py                   # 独立CLI脚本
├── config.json              # HF token存储（已gitignore）
├── requirements.txt         # Python依赖
├── speaker_output/          # 分离音频的输出目录
└── venv_portable/           # 虚拟环境
```

## 常用开发任务

### 测试说话人分离
将测试音频/视频放在项目根目录，运行:
```bash
python gradio_app.py
```
通过网页界面上传文件进行测试。

### 调试管道参数
在参数配置部分取消注释或添加print语句:
- `nodes.py:195` - 显示成功设置的参数
- `gradio_app.py:225` - Gradio界面的同样位置

### 添加新的输出模式
修改 `gradio_app.py:95-334` 中的 `SpeakerSeparator.separate_speakers()` 方法。该函数已经演示了两种输出模式的实现模式，可以扩展。

## 已知问题与限制

1. **视频处理:** 需要安装带ffmpeg后端的moviepy。检查 `gradio_app.py:13-32` 中的 `MOVIEPY_AVAILABLE` 标志和运行时检查
2. **最大说话人数:** ComfyUI节点限制为5个说话人输出（在RETURN_TYPES中硬编码）
3. **Token管理:** Token以明文存储在config.json中（已gitignore但未加密）
4. **内存使用:** 完整音频加载到内存；超长文件可能导致问题
5. **临时文件:** 管道需要基于文件的输入，会创建临时WAV文件并在finally块中清理

## 模型行为说明

- pyannote管道可能不会检测到指定的准确 `num_speakers` 数量；该参数仅作为提示
- 管道在清晰、不同的说话人情况下效果最好
- 背景噪音会影响分离质量
- 重叠的语音片段可能被归属到一个说话人
- 首次运行时模型会下载（约300MB）并缓存到本地

