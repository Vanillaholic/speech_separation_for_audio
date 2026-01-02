# ComfyUI 说话人分离节点

这是一个ComfyUI自定义节点，用于从音频中分离不同的说话人。

## 功能特点

- 输入音频对象，自动分离多个说话人
- 输出每个说话人的完整音频（非说话人时段为静音）
- 支持配置说话人数量、分离阈值等参数
- 支持保存Hugging Face Token以便后续使用

## 安装

1. 将此文件夹复制到ComfyUI的 `custom_nodes` 目录下

2. 安装依赖：
```bash
pip install pyannote.audio soundfile torch numpy
```

3. 在Hugging Face上申请访问权限：
   - 访问 https://huggingface.co/pyannote/speaker-diarization-3.1
   - 接受用户协议
   - 创建访问令牌：https://huggingface.co/settings/tokens

## 使用方法

1. 在ComfyUI中加载音频
2. 添加"说话人分离"节点（在 `audio/speaker_separation` 分类下）
3. 连接音频输入
4. 填写Hugging Face Token（首次使用需要）
5. 配置参数：
   - **说话人数量**：预期的说话人数量（1-10）
   - **Onset阈值**：语音开始检测阈值（0.0-1.0）
   - **Offset阈值**：语音结束检测阈值（0.0-1.0）
   - **最小语音时长**：最短语音片段时长（秒）
   - **最小静音时长**：最短静音片段时长（秒）
   - **聚类阈值**：说话人聚类阈值（0.0-1.0）
6. 运行节点，会输出最多5个说话人的音频（speaker_0 到 speaker_4）

## 节点参数说明

- **hf_token**: Hugging Face访问令牌，需要访问pyannote模型时使用
- **num_speakers**: 说话人数量，建议设置为实际的说话人数量
- **onset_threshold**: 语音开始检测的敏感度，值越大越严格
- **offset_threshold**: 语音结束检测的敏感度，值越大越严格
- **min_duration_on**: 过滤过短的语音片段（秒）
- **min_duration_off**: 过滤过短的静音片段（秒）
- **clustering_threshold**: 说话人聚类的阈值，影响说话人分组

## 输出说明

节点会输出5个音频输出端口：
- `speaker_0`: 第一个说话人的音频
- `speaker_1`: 第二个说话人的音频
- `speaker_2`: 第三个说话人的音频
- `speaker_3`: 第四个说话人的音频
- `speaker_4`: 第五个说话人的音频

如果检测到的说话人少于5个，多余的输出将是空音频（静音）。

每个说话人的音频文件长度与输入音频相同，其中非该说话人的时段为静音。

## 注意事项

1. 首次使用需要有效的Hugging Face Token
2. 处理较长音频时可能需要较长时间
3. 建议根据实际说话人数量设置 `num_speakers` 参数
4. 如果说话人数量设置不准确，可能需要调整聚类阈值

# speech_separation_for_audio
