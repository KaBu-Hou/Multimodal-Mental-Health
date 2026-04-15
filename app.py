import gradio as gr
import torch
import os
import numpy as np

# 导入你的模型结构和数据处理工具
from chsims_dataset import CHSIMSDataset 
from multimodal_student import StudentMultimodalModel, ModelConfig
from transformers import AutoTokenizer

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("🚀 正在初始化 Web 演示系统...")

# ==========================================
# 1. 挂载最强模型权重 (V3版本)
# ==========================================
config = ModelConfig()
model = StudentMultimodalModel(config=config, bert_model_path='./bert-base-chinese').to(device)

ckpt_path = 'checkpoints/best_model.pth'
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ 最佳模型权重加载成功 (V3 Golden Checkpoint)！")
else:
    print("❌ 找不到权重文件，请确认 checkpoints/best_model.pth 是否存在。")

model.eval()

# ==========================================
# 2. 初始化预处理工具
# ==========================================
# 实例化 Dataset 仅仅是为了借用里面的特征提取函数 (抽帧、提取 MFCC/Mel)
# 注意：这里的 data_dir 填你原始 SIMS 视频所在的根目录
dummy_dataset = CHSIMSDataset(data_dir='../../SIMS', split='test') 
tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese')

# ==========================================
# 3. 核心推理函数
# ==========================================
def predict_risk(video_file, text_input):
    if video_file is None or text_input.strip() == "":
        return "⚠️ 缺少数据：请同时提供视频和对应的文本内容！", 0.0
    
    video_path = video_file
    
    try:
        # --- 1. 动态特征提取 ---
        # 提取视频帧 (16, 3, 224, 224)
        video_frames = dummy_dataset.extract_video_frames(video_path).unsqueeze(0).to(device)
        
        # 提取音频特征
        audio_features = dummy_dataset.extract_audio_features_safe(video_path)
        audio_mfcc = torch.tensor(audio_features['mfcc'], dtype=torch.float32).unsqueeze(0).to(device)
        audio_mel = torch.tensor(audio_features['mel_spec'], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 处理文本特征
        text_encoding = tokenizer(
            text_input, max_length=128,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        text_ids = text_encoding['input_ids'].to(device)
        attention_mask = text_encoding['attention_mask'].to(device)
        
        # --- 2. 模型前向推理 ---
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                outputs = model(text_ids, attention_mask, video_frames, audio_mfcc, audio_mel)
                
        # --- 3. 解析输出结果 ---
        emotion_score = outputs['emotion_pred'][0].cpu().numpy().tolist()
        risk_score_raw = outputs['risk_scores'][0].cpu().numpy().tolist()
        pred_level = outputs['risk_level'].argmax(dim=1).item()
        
        # 风险等级映射
        level_map = {
            0: "🟢 低风险 (情绪较稳定，无需干预)",
            1: "🟡 中风险 (出现波动，建议关注疏导)",
            2: "🔴 高风险 (情绪极值，建议立刻人工介入)"
        }
        
        # 综合情感波动绝对值 (用于进度条展示)
        avg_emotion_volatility = float(np.mean(np.abs(emotion_score)))
        
        # 格式化诊断报告
        result_text = f"""
### 🎯 综合诊断结果：{level_map[pred_level]}

**📊 多模态底层指标拆解：**
* **连续情感波动打分 (越接近0越平稳):** `{avg_emotion_volatility:.3f}`
* **三模态独立风险概率 (文本/音频/视觉):** `{[round(s, 3) for s in risk_score_raw]}`

*(注：系统通过捕捉微表情、声音频率震颤以及负面语义，综合给出了上述评判。)*
        """
        return result_text, avg_emotion_volatility

    except Exception as e:
        return f"❌ 处理过程中发生错误：{str(e)}", 0.0

# ==========================================
# 4. 构建 Gradio Web 交互界面
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 大学生多模态心理健康检测系统 (V3 终极版)")
    gr.Markdown("> 融合 **自然语言 (BERT)** + **语音频域 (MFCC/Mel)** + **面部微表情 (CNN3D)** 三重模态，精准捕捉隐性心理风险。")
    
    with gr.Row():
        # 左侧：输入区
        with gr.Column(scale=1):
            video_in = gr.Video(label="上传被访者视频 (MP4格式)")
            text_in = gr.Textbox(label="对应的文本内容 (视频里说了什么？)", lines=4, placeholder="例如：我最近总是失眠，感觉压力很大，对什么都提不起兴趣...")
            submit_btn = gr.Button("🧠 启动多模态融合分析", variant="primary")
            
        # 右侧：结果展示区
        with gr.Column(scale=1):
            result_display = gr.Markdown("等待输入数据...")
            # 情感波动仪表盘
            emotion_gauge = gr.Slider(minimum=0, maximum=2.0, label="系统测算：总体情感波动烈度", interactive=False)

    # 绑定点击事件
    submit_btn.click(
        fn=predict_risk,
        inputs=[video_in, text_in],
        outputs=[result_display, emotion_gauge]
    )

if __name__ == "__main__":
    # share=True 自动生成公网访问链接
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)