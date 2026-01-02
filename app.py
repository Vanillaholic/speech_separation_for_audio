import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
import soundfile as sf
from pathlib import Path
import os

def separate_speakers_to_files(audio_file, output_dir="./output", hf_token="YOUR_HF_TOKEN_HERE"):
    """
    分离音频中的说话人，并为每个说话人生成独立的音频文件
    
    参数:
        audio_file: 输入音频文件路径
        output_dir: 输出目录
        hf_token: Hugging Face token
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 步骤1: 设置 Hugging Face token 为环境变量
    os.environ["HF_TOKEN"] = hf_token
    
    # 步骤2: 加载预训练的说话人分离管道（使用新的认证方式）
    print("加载说话人分离模型...")
    try:
        # 方法1: 使用 token 参数
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
    except Exception as e:
        print(f"方法1失败: {e}")
        try:
            # 方法2: 使用 use_auth_token 参数（某些版本可能还支持）
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",#speaker-diarization-community-1
                use_auth_token=hf_token
            )
        except Exception as e2:
            print(f"方法2也失败: {e2}")
            # 方法3: 尝试不使用 token（如果模型已缓存）
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
            except Exception as e3:
                print(f"所有方法都失败: {e3}")
                print("请检查:")
                print("1. Hugging Face Token 是否正确")
                print("2. 是否已在 Hugging Face 上同意 pyannote/speaker-diarization-3.1 的用户协议")
                print("3. 网络连接是否正常")
                return
    
    # ========== 查看并调整管道参数 ==========
    print("查看管道可用参数...")
    
    # 首先查看管道支持哪些参数
    try:
        # 获取管道的默认参数
        default_params = pipeline.parameters(instantiated=True)
        print("管道支持的参数:", list(default_params.keys()))
    except Exception as e:
        print(f"获取参数失败: {e}")
        default_params = {}
    
    # ========== 尝试不同的参数设置方法 ==========
    print("调整管道参数以优化纯净度...")
    
    try:
        # 方法1: 尝试使用正确的参数名
        # 对于说话人日志任务，通常的参数名可能是这些
        param_candidates = {
            # 语音活动检测相关参数
            'onset': 0.7, 'offset': 0.7,
            'min_duration_on': 0.1, 'min_duration_off': 0.1,
            'stitch_threshold': 0.04,
            
            # 聚类相关参数
            'clustering': {'method': 'centroid', 'min_cluster_size': 12},
            
            # 说话人嵌入相关参数
            'embedding': {'exclude_center': True}
        }
        
        # 只设置存在的参数
        valid_params = {}
        for param_name, param_value in param_candidates.items():
            if param_name in default_params:
                valid_params[param_name] = param_value
                print(f"设置参数 {param_name} = {param_value}")
            elif isinstance(param_value, dict):
                # 处理嵌套字典参数
                for sub_param, sub_value in param_value.items():
                    full_param_name = f"{param_name}.{sub_param}"
                    if full_param_name in default_params:
                        valid_params[full_param_name] = sub_value
                        print(f"设置参数 {full_param_name} = {sub_value}")
        
        if valid_params:
            pipeline.instantiate(valid_params)
            print("参数设置成功！")
        else:
            print("未找到可用的参数，使用默认参数")
            
    except Exception as e:
        print(f"参数设置失败: {e}")
        print("将继续使用默认参数...")
    
    # ========== 备选方案：直接修改实例化后的组件 ==========
    try:
        # 查看管道内部的组件
        if hasattr(pipeline, 'segmentation'):
            print("管道包含 segmentation 组件")
        if hasattr(pipeline, 'embedding'):
            print("管道包含 embedding 组件") 
        if hasattr(pipeline, 'clustering'):
            print("管道包含 clustering 组件")
    except:
        pass
    # 步骤3: 读取原始音频文件
    print(f"读取音频文件: {audio_file}")
    try:
        audio_data, sample_rate = sf.read(audio_file)
    except Exception as e:
        print(f"读取音频文件失败: {e}")
        return
    
    # 确保音频是单声道（如果不是，转换为单声道）
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        print("已将立体声音频转换为单声道")
    
    # 步骤4: 应用管道进行说话人分离
    print("正在进行说话人分离...")
    try:
        diarization = pipeline(audio_file,num_speakers=2)
    except Exception as e:
        print(f"说话人分离失败: {e}")
        return
    
    # 收集所有说话人的时间段
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"说话人 {speaker}: 从 {turn.start:.1f}秒 开始，到 {turn.end:.1f}秒 结束")
        
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))
    
    # 如果没有检测到说话人
    if not speaker_segments:
        print("未检测到任何说话人！")
        return
    
    # 步骤5: 为每个说话人创建完整的音频文件
    print("\n生成各说话人的独立音频文件...")
    total_duration = len(audio_data) / sample_rate
    
    for speaker, segments in speaker_segments.items():
        print(f"处理说话人 {speaker}...")
        
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
        
        # 保存说话人的音频文件
        output_file = output_path / f"speaker_{speaker}.wav"
        sf.write(output_file, speaker_audio, sample_rate)
        print(f"  已保存: {output_file}")
    
    # 步骤6: 生成一个包含所有说话人活动的时间线文件
    timeline_file = output_path / "speaker_timeline.txt"
    with open(timeline_file, 'w', encoding='utf-8') as f:
        f.write(f"音频文件: {audio_file}\n")
        f.write(f"总时长: {total_duration:.2f}秒\n")
        f.write(f"采样率: {sample_rate}Hz\n")
        f.write(f"检测到的说话人数量: {len(speaker_segments)}\n")
        f.write("\n说话人时间线:\n")
        f.write("-" * 50 + "\n")
        
        for speaker, segments in speaker_segments.items():
            total_speaker_time = sum(end - start for start, end in segments)
            f.write(f"\n说话人 {speaker} (总发言时间: {total_speaker_time:.2f}s):\n")
            for i, (start, end) in enumerate(segments, 1):
                f.write(f"  片段 {i}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)\n")
    
    print(f"\n处理完成！")
    print(f"输出目录: {output_path.absolute()}")
    print(f"时间线文件: {timeline_file}")
    print(f"共检测到 {len(speaker_segments)} 个说话人")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径和Hugging Face Token
    input_audio = "input.flac"  # 请替换为你的音频文件路径
    hf_token = "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"  # 请替换为你的Hugging Face Token
    
    # 运行说话人分离
    separate_speakers_to_files(
        audio_file=input_audio,
        output_dir="./speaker_output",
        hf_token=hf_token
    )
