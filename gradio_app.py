import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
import soundfile as sf
from pathlib import Path
import os
import tempfile
import json
import gradio as gr

# 视频处理
MOVIEPY_AVAILABLE = False
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    # ImportError: 模块未安装或找不到
    error_msg = str(e)
    if "No module named 'moviepy'" in error_msg or "No module named 'moviepy.editor'" in error_msg:
        pass  # 只在真正缺少模块时打印，否则不打印（可能在运行时可用）
except Exception:
    pass  # 静默处理其他异常，在运行时再检查

# 在运行时检查 moviepy 是否可用
def check_moviepy_available():
    """运行时检查 moviepy 是否可用"""
    try:
        from moviepy.editor import VideoFileClip
        return True
    except:
        return False

class SpeakerSeparator:
    """音频说话人分离器，支持两种输出模式"""
    
    def __init__(self, config_file="config.json"):
        self.pipeline_cache = None
        self.cached_token = None
        self.config_file = config_file
        self.hf_token = self.load_token_from_config()
    
    def load_token_from_config(self):
        """从配置文件读取 token"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    token = config.get('hf_token', '')
                    if token and token != "请输入你的 Hugging Face Token":
                        return token
            print(f"警告: 未找到有效的 token，请检查 {self.config_file}")
            return None
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            return None
    
    def load_pipeline(self, hf_token):
        """加载或获取缓存的pipeline"""
        if self.pipeline_cache is None or self.cached_token != hf_token:
            if not hf_token:
                raise ValueError("请输入有效的Hugging Face Token")
            
            os.environ["HF_TOKEN"] = hf_token
            
            try:
                self.pipeline_cache = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
            except Exception as e:
                try:
                    self.pipeline_cache = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                except Exception as e2:
                    try:
                        self.pipeline_cache = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1"
                        )
                    except Exception as e3:
                        raise ValueError(
                            f"无法加载模型: {e3}\n"
                            "请检查:\n"
                            "1. Hugging Face Token 是否正确\n"
                            "2. 是否已在 Hugging Face 上同意 pyannote/speaker-diarization-3.1 的用户协议\n"
                            "3. 网络连接是否正常"
                        )
            
            self.cached_token = hf_token
        
        return self.pipeline_cache
    
    def separate_speakers(
        self,
        audio_file,
        num_speakers,
        onset_threshold,
        offset_threshold,
        min_duration_on,
        min_duration_off,
        clustering_threshold,
        output_mode1,
        output_mode2
    ):
        """
        执行说话人分离
        
        参数:
            audio_file: 音频文件路径（Gradio上传的文件）
            hf_token: Hugging Face token
            num_speakers: 说话人数量
            onset_threshold: 语音开始阈值
            offset_threshold: 语音结束阈值
            min_duration_on: 最小语音时长
            min_duration_off: 最小静音时长
            clustering_threshold: 聚类阈值
            output_mode1: 是否输出模式1（时长=原音频，非说话人时段为静音）
            output_mode2: 是否输出模式2（只包含说话时段，无静音）
        
        返回:
            输出文件列表和消息
        """
        
        if audio_file is None:
            return [], [], "请上传音频或视频文件"
        
        # 使用配置中的 token
        hf_token = self.hf_token
        if not hf_token:
            return [], [], f"错误: 未找到有效的 Hugging Face Token，请检查 {self.config_file} 配置文件"
        
        # 初始化消息
        message = ""
        
        # 检查文件类型，如果是视频则先提取音频
        file_ext = Path(audio_file).suffix.lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        
        actual_audio_file = audio_file
        temp_audio_path = None
        
        if is_video:
            # 运行时再次检查 moviepy 是否可用
            try:
                from moviepy.editor import VideoFileClip
            except ImportError as e:
                return [], [], f"错误: 无法处理视频文件。moviepy 导入失败: {e}\n请检查 moviepy 是否正确安装: pip install moviepy"
            except Exception as e:
                return [], [], f"错误: 无法处理视频文件。moviepy 导入时出现异常: {type(e).__name__}: {e}\n如果已安装 moviepy，可能是依赖问题，请检查 imageio-ffmpeg: pip install imageio-ffmpeg"
            
            try:
                message += "检测到视频文件，正在提取音频...\n"
                # 从视频提取音频
                with VideoFileClip(audio_file) as video:
                    audio = video.audio
                    if audio is None:
                        return [], [], "错误: 视频文件中没有音频轨道"
                    
                    # 创建临时音频文件
                    temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_audio_path = temp_audio_file.name
                    temp_audio_file.close()
                    
                    # 写入音频
                    audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                    actual_audio_file = temp_audio_path
                    message += f"音频提取完成，采样率: {audio.fps}Hz\n"
                    # 关闭音频对象以释放资源
                    audio.close()
            except Exception as e:
                return [], [], f"从视频提取音频失败: {e}"
        
        # 读取音频文件
        try:
            audio_data, sample_rate = sf.read(actual_audio_file)
        except Exception as e:
            return [], [], f"读取音频文件失败: {e}"
        finally:
            # 清理临时音频文件
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
        
        # 确保音频是单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 转换为float32
        audio_data = audio_data.astype(np.float32)
        
        # 加载pipeline
        try:
            pipeline = self.load_pipeline(hf_token)
        except Exception as e:
            return [], [], str(e)
        
        # 配置pipeline参数
        try:
            default_params = pipeline.parameters(instantiated=True)
            valid_params = {}
            
            param_map = {
                'onset': onset_threshold,
                'offset': offset_threshold,
                'min_duration_on': min_duration_on,
                'min_duration_off': min_duration_off,
            }
            
            for param_name, param_value in param_map.items():
                if param_name in default_params:
                    valid_params[param_name] = param_value
            
            # 尝试设置聚类参数
            if 'clustering' in default_params:
                if isinstance(default_params['clustering'], dict):
                    clustering_params = default_params['clustering'].copy()
                    clustering_params['threshold'] = clustering_threshold
                    valid_params['clustering'] = clustering_params
            
            if valid_params:
                pipeline.instantiate(valid_params)
        except Exception as e:
            print(f"参数设置失败，使用默认参数: {e}")
        
        # 创建临时文件用于pipeline处理
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sample_rate)
        
        output_files = []
        audio_list = []
        
        # message 已经在前面初始化了，如果有视频处理则已有内容
        
        try:
            # 执行说话人分离
            diarization = pipeline(tmp_path, num_speakers=num_speakers)
            
            # 收集所有说话人的时间段
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((turn.start, turn.end))
            
            if not speaker_segments:
                message = "未检测到任何说话人！"
                return output_files, [], message
            
            # 按说话人ID排序
            sorted_speakers = sorted(speaker_segments.keys())
            num_detected = len(sorted_speakers)
            message += f"检测到 {num_detected} 个说话人\n"
            
            # 创建输出目录
            output_dir = Path('speaker_output')
            output_dir.mkdir(exist_ok=True)
            
            # 模式1：时长等于原音频，非说话人时段为静音
            if output_mode1:
                message += "\n模式1输出（时长=原音频，非说话人时段为静音）：\n"
                for speaker_id in sorted_speakers:
                    segments = speaker_segments[speaker_id]
                    
                    # 创建全零数组（静音），长度与原始音频相同
                    speaker_audio = np.zeros_like(audio_data)
                    
                    # 将说话人的语音段复制到对应位置
                    for start_time, end_time in segments:
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        
                        # 确保索引不越界
                        start_sample = min(start_sample, len(audio_data))
                        end_sample = min(end_sample, len(audio_data))
                        
                        if start_sample < end_sample:
                            speaker_audio[start_sample:end_sample] = audio_data[start_sample:end_sample]
                    
                    # 保存文件
                    output_file = output_dir / f"mode1_speaker_{speaker_id}.wav"
                    sf.write(str(output_file), speaker_audio, sample_rate)
                    output_files.append(str(output_file))
                    message += f"  - {output_file.name}\n"
            
            # 模式2：只包含说话时段，无静音，时长不等于原音频
            if output_mode2:
                message += "\n模式2输出（只包含说话时段，无静音）：\n"
                for speaker_id in sorted_speakers:
                    segments = speaker_segments[speaker_id]
                    
                    # 收集所有该说话人的音频片段
                    speaker_chunks = []
                    for start_time, end_time in segments:
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        
                        # 确保索引不越界
                        start_sample = min(start_sample, len(audio_data))
                        end_sample = min(end_sample, len(audio_data))
                        
                        if start_sample < end_sample:
                            chunk = audio_data[start_sample:end_sample]
                            speaker_chunks.append(chunk)
                    
                    # 如果有音频片段，拼接它们
                    if speaker_chunks:
                        speaker_audio = np.concatenate(speaker_chunks)
                        
                        # 保存文件
                        output_file = output_dir / f"mode2_speaker_{speaker_id}.wav"
                        sf.write(str(output_file), speaker_audio, sample_rate)
                        output_files.append(str(output_file))
                        message += f"  - {output_file.name}\n"
            
            message += f"\n处理完成！共生成 {len(output_files)} 个文件。"
            
        except Exception as e:
            message = f"处理失败: {str(e)}"
        
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # 返回文件列表和消息
        # output_files 是文件路径列表，用于显示和下载
        return output_files, output_files, message


# 创建分离器实例
separator = SpeakerSeparator()

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="音视频说话人分离工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 音视频说话人分离工具 | authorized by Zihan Xing")
        gr.Markdown("上传音频或视频文件，自动分离不同的说话人。支持两种输出模式：")
        gr.Markdown("- **模式1**：每个说话人的音频时长等于原音频，非说话人时段为静音")
        gr.Markdown("- **模式2**：每个说话人的音频只包含说话时段，无静音，时长不等于原音频")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.File(
                    label="上传音频或视频文件",
                    file_types=["audio", "video"]
                )
                
                # 添加一个音频预览组件（仅用于预览上传的文件）
                audio_preview = gr.Audio(
                    label="音频预览（上传后自动显示）",
                    type="filepath",
                    interactive=False
                )
                
                with gr.Accordion("分离参数设置", open=False):
                    num_speakers = gr.Slider(
                        label="说话人数量",
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        info="预期的说话人数量"
                    )
                    
                    onset_threshold = gr.Slider(
                        label="Onset阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.01,
                        info="语音开始检测的敏感度，值越大越严格"
                    )
                    
                    offset_threshold = gr.Slider(
                        label="Offset阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.01,
                        info="语音结束检测的敏感度，值越大越严格"
                    )
                    
                    min_duration_on = gr.Slider(
                        label="最小语音时长(秒)",
                        minimum=0.0,
                        maximum=5.0,
                        value=0.1,
                        step=0.01,
                        info="过滤过短的语音片段"
                    )
                    
                    min_duration_off = gr.Slider(
                        label="最小静音时长(秒)",
                        minimum=0.0,
                        maximum=5.0,
                        value=0.1,
                        step=0.01,
                        info="过滤过短的静音片段"
                    )
                    
                    clustering_threshold = gr.Slider(
                        label="聚类阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.01,
                        info="说话人聚类的阈值，影响说话人分组"
                    )
                
                with gr.Accordion("输出模式选择", open=True):
                    output_mode1 = gr.Checkbox(
                        label="模式1：时长等于原音频（非说话人时段为静音）",
                        value=True,
                        info="每个说话人的音频时长等于原音频时长"
                    )
                    
                    output_mode2 = gr.Checkbox(
                        label="模式2：只包含说话时段（无静音）",
                        value=False,
                        info="每个说话人的音频只包含说话时段，时长不等于原音频"
                    )
                    
                    gr.Markdown("⚠️ 至少选择一种输出模式")
                
                process_btn = gr.Button("开始分离", variant="primary", size="lg")
            
            with gr.Column():
                message_output = gr.Textbox(
                    label="处理信息",
                    lines=10,
                    interactive=False
                )
                
                # 输出音频预览区域
                gr.Markdown("### 输出音频文件（可预览和下载）")
                
                with gr.Row():
                    output_audio_1 = gr.Audio(label="说话人 1", type="filepath")
                    output_audio_2 = gr.Audio(label="说话人 2", type="filepath")
                
                with gr.Row():
                    output_audio_3 = gr.Audio(label="说话人 3", type="filepath")
                    output_audio_4 = gr.Audio(label="说话人 4", type="filepath")
                
                output_audio_5 = gr.Audio(label="说话人 5", type="filepath")
        
        # 处理函数
        def process_audio(
            audio_file,
            num_speakers,
            onset_threshold,
            offset_threshold,
            min_duration_on,
            min_duration_off,
            clustering_threshold,
            output_mode1,
            output_mode2
        ):
            if not output_mode1 and not output_mode2:
                return (
                    "请至少选择一种输出模式！",
                    None,  # 预览音频
                    None,
                    None,
                    None,
                    None,
                    None
                )
            
            # gr.File 返回文件路径（可能是字符串或文件对象）
            audio_path = None
            preview_audio_path = None
            
            if audio_file is not None:
                if isinstance(audio_file, (list, tuple)) and len(audio_file) > 0:
                    # 如果是列表，取第一个
                    audio_path = audio_file[0].name if hasattr(audio_file[0], 'name') else str(audio_file[0])
                elif hasattr(audio_file, 'name'):
                    # 文件对象有 name 属性
                    audio_path = audio_file.name
                else:
                    # 直接是字符串路径
                    audio_path = str(audio_file)
                
                # 如果是视频文件，先提取音频用于预览
                file_ext = Path(audio_path).suffix.lower() if audio_path else ""
                is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
                
                if is_video:
                    # 对于视频，提取音频用于预览
                    if check_moviepy_available():
                        try:
                            from moviepy.editor import VideoFileClip
                            with VideoFileClip(audio_path) as video:
                                if video.audio:
                                    temp_preview = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                    temp_preview_path = temp_preview.name
                                    temp_preview.close()
                                    video.audio.write_audiofile(temp_preview_path, verbose=False, logger=None)
                                    preview_audio_path = temp_preview_path
                                    video.audio.close()
                        except Exception as e:
                            print(f"视频预览失败: {e}")
                else:
                    # 如果是音频文件，直接使用
                    preview_audio_path = audio_path
            
            files, audio_list, message = separator.separate_speakers(
                audio_file=audio_path,
                num_speakers=int(num_speakers),
                onset_threshold=onset_threshold,
                offset_threshold=offset_threshold,
                min_duration_on=min_duration_on,
                min_duration_off=min_duration_off,
                clustering_threshold=clustering_threshold,
                output_mode1=output_mode1,
                output_mode2=output_mode2
            )
            
            # 将文件列表添加到消息中
            if files:
                message += "\n\n" + "="*50 + "\n"
                message += "生成的文件列表：\n"
                message += "="*50 + "\n"
                for i, file_path in enumerate(files, 1):
                    message += f"{i}. {file_path}\n"
            
            # 准备返回的音频文件（最多5个）
            audio_outputs = [None] * 5
            for i, file_path in enumerate(files[:5]):
                audio_outputs[i] = file_path
            
            # 返回消息、预览音频和输出音频文件（显示前5个）
            # 在 Gradio 4.19.0 中直接返回值，而不是使用 update() 方法
            return (
                message,
                preview_audio_path,  # 添加预览音频
                audio_outputs[0],
                audio_outputs[1],
                audio_outputs[2],
                audio_outputs[3],
                audio_outputs[4]
            )
        
        # 文件上传后的预览函数
        def preview_uploaded_file(audio_file):
            """当文件上传后，自动显示预览"""
            if audio_file is None:
                return None
            
            # 获取文件路径
            if isinstance(audio_file, (list, tuple)) and len(audio_file) > 0:
                audio_path = audio_file[0].name if hasattr(audio_file[0], 'name') else str(audio_file[0])
            elif hasattr(audio_file, 'name'):
                audio_path = audio_file.name
            else:
                audio_path = str(audio_file)
            
            if not audio_path:
                return None
            
            # 检查是否为视频文件
            file_ext = Path(audio_path).suffix.lower()
            is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
            
            if is_video:
                # 如果是视频，提取音频用于预览
                if not check_moviepy_available():
                    # 不打印警告，只是返回 None
                    return None
                try:
                    from moviepy.editor import VideoFileClip
                    with VideoFileClip(audio_path) as video:
                        if video.audio:
                            temp_preview = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_preview_path = temp_preview.name
                            temp_preview.close()
                            video.audio.write_audiofile(temp_preview_path, verbose=False, logger=None)
                            video.audio.close()
                            return temp_preview_path
                except Exception as e:
                    print(f"视频预览失败: {e}")
                    return None
            else:
                # 如果是音频文件，直接返回路径
                return audio_path
        
        # 绑定文件上传事件（自动预览）
        audio_input.change(
            fn=preview_uploaded_file,
            inputs=[audio_input],
            outputs=[audio_preview]
        )
        
        # 绑定处理事件
        process_btn.click(
            fn=process_audio,
            inputs=[
                audio_input,
                num_speakers,
                onset_threshold,
                offset_threshold,
                min_duration_on,
                min_duration_off,
                clustering_threshold,
                output_mode1,
                output_mode2
            ],
            outputs=[
                message_output,
                audio_preview,
                output_audio_1,
                output_audio_2,
                output_audio_3,
                output_audio_4,
                output_audio_5
            ]
        )
    
    return demo


if __name__ == "__main__":
    import socket
    import os
    
    # 禁用 Gradio 的 analytics 以避免网络超时错误
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    def find_free_port(start_port=7860, max_attempts=10):
        """查找可用端口"""
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None
    
    demo = create_interface()
    
    # 尝试找到可用端口
    port = find_free_port(7860)
    if port:
        print(f"正在启动服务，端口: {port}")
        try:
            demo.launch(server_port=port, share=False, show_error=True)
        except OSError as e:
            if "Cannot find empty port" in str(e):
                print(f"端口 {port} 不可用，尝试自动选择端口...")
                demo.launch(share=False, show_error=True, server_port=None)
            else:
                raise
    else:
        print("正在启动服务（自动选择端口）")
        demo.launch(share=False, show_error=True, server_port=None)


