import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
import soundfile as sf
from pathlib import Path
import os
import tempfile

class SpeakerSeparationNode:
    """
    ComfyUI节点：说话人分离
    输入：音频对象
    输出：各个分离的说话人音频文件列表
    """
    
    def __init__(self):
        self.pipeline_cache = None
        self.cached_token = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "display_name": "Hugging Face Token"
                }),
                "num_speakers": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display_name": "说话人数量"
                }),
                "onset_threshold": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display_name": "Onset阈值"
                }),
                "offset_threshold": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display_name": "Offset阈值"
                }),
                "min_duration_on": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "display_name": "最小语音时长(秒)"
                }),
                "min_duration_off": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "display_name": "最小静音时长(秒)"
                }),
                "clustering_threshold": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display_name": "聚类阈值"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("speaker_0", "speaker_1", "speaker_2", "speaker_3", "speaker_4")
    FUNCTION = "separate"
    CATEGORY = "audio/speaker_separation"
    
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
            print("说话人分离模型加载成功")
        
        return self.pipeline_cache
    
    def separate(self, audio, hf_token, num_speakers, onset_threshold, 
                 offset_threshold, min_duration_on, min_duration_off, clustering_threshold):
        """
        执行说话人分离
        
        参数:
            audio: ComfyUI音频对象 (通常是元组或字典，包含音频数据和采样率)
            hf_token: Hugging Face token
            num_speakers: 说话人数量
            onset_threshold: 语音开始阈值
            offset_threshold: 语音结束阈值
            min_duration_on: 最小语音时长
            min_duration_off: 最小静音时长
            clustering_threshold: 聚类阈值
        """
        
        # 解析ComfyUI音频输入
        # ComfyUI的AUDIO类型通常是 (audio_tensor, sample_rate) 元组
        if isinstance(audio, tuple) and len(audio) == 2:
            audio_data, sample_rate = audio
        elif isinstance(audio, dict):
            audio_data = audio.get("samples", audio.get("audio"))
            sample_rate = audio.get("sample_rate", 16000)
        else:
            # 尝试直接作为音频数据
            audio_data = audio
            sample_rate = 16000  # 默认采样率
        
        # 转换为numpy数组
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        elif not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # 确保是一维数组（shape可能是多维）
        if audio_data.ndim > 1:
            # 如果是立体声，转换为单声道
            if audio_data.shape[0] < audio_data.shape[-1]:
                # 假设最后一个维度是通道
                audio_data = np.mean(audio_data, axis=-1)
            else:
                # 假设第一个维度是通道
                audio_data = np.mean(audio_data, axis=0)
        
        # 确保是一维数组
        audio_data = audio_data.flatten()
        audio_data = audio_data.astype(np.float32)
        
        # 加载pipeline
        pipeline = self.load_pipeline(hf_token)
        
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
                print(f"已设置参数: {valid_params}")
        except Exception as e:
            print(f"参数设置失败，使用默认参数: {e}")
        
        # 创建临时文件用于pipeline处理
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sample_rate)
        
        try:
            # 执行说话人分离
            print(f"正在进行说话人分离，预计说话人数量: {num_speakers}")
            diarization = pipeline(tmp_path, num_speakers=num_speakers)
            
            # 收集所有说话人的时间段
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((turn.start, turn.end))
            
            if not speaker_segments:
                print("未检测到任何说话人！")
                # 返回空音频
                empty_audio = np.zeros_like(audio_data)
                empty_tuple = (empty_audio, sample_rate)
                return (empty_tuple,) * 5
            
            output_dir = 'speaker_output'
            os.makedirs(output_dir, exist_ok=True)

            # 为每个说话人创建完整的音频文件
            speaker_audios = []
            total_duration = len(audio_data) / sample_rate
            
            # 按说话人ID排序
            sorted_speakers = sorted(speaker_segments.keys())
            
            for idx, speaker_id in enumerate(sorted_speakers[:5]):  # 最多支持5个说话人
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
                
                tensor_audio = torch.from_numpy(speaker_audio.astype(np.float32))  # 保证float32
                if tensor_audio.dim() == 1:
                    tensor_audio = tensor_audio.unsqueeze(0)  # 保证 [1, T]
                speaker_audios.append((tensor_audio, sample_rate))
            
            # 如果说话人数量少于5个，用空音频填充
            while len(speaker_audios) < 5:
                idx = len(speaker_audios)
                empty_audio = np.zeros_like(audio_data)
                tensor_audio = torch.from_numpy(empty_audio.astype(np.float32))
                if tensor_audio.dim() == 1:
                    tensor_audio = tensor_audio.unsqueeze(0)
                speaker_audios.append((tensor_audio, sample_rate))
            
            print(f"说话人分离完成，检测到 {len(sorted_speakers)} 个说话人 (torch.Tensor输出)")
            return speaker_audios[0], speaker_audios[1], speaker_audios[2], speaker_audios[3], speaker_audios[4]
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass

class SpeakerSeparationSingleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {"default": "", "multiline": False, "display_name": "Hugging Face Token"}),
                "num_speakers": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1, "display_name": "说话人数量"}),
                "onset_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display_name": "Onset阈值"}),
                "offset_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display_name": "Offset阈值"}),
                "min_duration_on": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01, "display_name": "最小语音时长(秒)"}),
                "min_duration_off": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01, "display_name": "最小静音时长(秒)"}),
                "clustering_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display_name": "聚类阈值"}),
                "speaker_index": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, "display_name": "预览说话人索引(0-4)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "separate_single"
    CATEGORY = "audio/speaker_separation"

    def separate_single(self, audio, hf_token, num_speakers, onset_threshold,
                        offset_threshold, min_duration_on, min_duration_off, clustering_threshold, speaker_index):
        base = SpeakerSeparationNode()
        outputs = base.separate(audio, hf_token, num_speakers, onset_threshold,
                                offset_threshold, min_duration_on, min_duration_off, clustering_threshold)
        # 保护性处理
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            raise ValueError("说话人分离结果为空，无法预览")
        idx = max(0, min(int(speaker_index), len(outputs) - 1))
        return (outputs[idx],)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "SpeakerSeparationNode": SpeakerSeparationNode,
    "SpeakerSeparationSingleNode": SpeakerSeparationSingleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerSeparationNode": "说话人分离",
    "SpeakerSeparationSingleNode": "说话人分离(单路预览)",
}

