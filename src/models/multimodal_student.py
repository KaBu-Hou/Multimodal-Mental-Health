import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class StudentMultimodalModel(nn.Module):
    
    def __init__(self, config, bert_model_path='../bert-base-chinese'):
        super().__init__()
        self.config = config
        
        # 1. 文本编码器
        print(f"加载本地BERT模型: {bert_model_path}")
        try:
            self.text_encoder = AutoModel.from_pretrained(bert_model_path)
            print("✓ 本地BERT模型加载成功")
        except Exception as e:
            print(f"加载本地BERT模型失败: {e}")
            self.text_encoder = AutoModel.from_pretrained('bert-base-chinese')
        
        text_hidden_size = self.text_encoder.config.hidden_size
        if config.freeze_text_layers > 0:
            for param in self.text_encoder.embeddings.parameters():
                param.requires_grad = False
            for i in range(config.freeze_text_layers):
                for param in self.text_encoder.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        # 文本投影层
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.visual_projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. 音频编码器
        self.audio_mfcc_encoder = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32)
        )
        
        self.audio_mel_encoder = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32)
        )
        
        self.audio_lstm = nn.LSTM(
            input_size=256 + 256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.audio_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 4. 多模态融合
        self.modal_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # 🔥 修复2：融合层输入维度改为 1024
        p = self.config.dropout_prob
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, 512),  # 256(注意力) + 256*3(原始) = 1024
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p)
        )
        self.emotion_regressor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 3)
        )
        
        self.mental_risk_assessor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 3)
        )
        
        self.risk_classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),nn.Dropout(p),
            nn.Linear(32, 3)
        )
    
    def forward(self, text_input_ids, text_attention_mask, video_frames, audio_mfcc, audio_mel):
        batch_size = text_input_ids.size(0)
        
        # 1. 文本特征
        text_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        
        # 2. 视觉特征
        video_input = video_frames.permute(0, 2, 1, 3, 4)
        visual_features = self.visual_encoder(video_input)
        visual_features = visual_features.view(batch_size, -1)
        visual_features = self.visual_projection(visual_features)
        
        # 3. 音频特征
        mfcc_input = audio_mfcc.transpose(1, 2) if audio_mfcc.dim() == 3 else audio_mfcc
        mfcc_features = self.audio_mfcc_encoder(mfcc_input).transpose(1, 2)
        
        mel_input = audio_mel.transpose(1, 2) if audio_mel.dim() == 3 else audio_mel
        mel_features = self.audio_mel_encoder(mel_input).transpose(1, 2)
        
        audio_combined = torch.cat([mfcc_features, mel_features], dim=2)
        audio_features, _ = self.audio_lstm(audio_combined)
        audio_features = torch.mean(audio_features, dim=1)
        audio_features = self.audio_projection(audio_features)
        
        # ========================
        # 🔥 修复3：启用模态注意力融合
        # ========================
        modal_features = torch.stack([text_features, visual_features, audio_features], dim=1)
        attended_features, _ = self.modal_attention(modal_features, modal_features, modal_features)
        attended_features_pooled = attended_features.mean(dim=1)
        
        # 融合：注意力特征 + 原始特征
        combined_features = torch.cat([
            attended_features_pooled,
            text_features, visual_features, audio_features
        ], dim=1)
        
        fused_features = self.fusion_layer(combined_features)
        
        # 5. 多任务输出
        emotion_pred = self.emotion_regressor(fused_features)
        risk_scores = self.mental_risk_assessor(fused_features)
        risk_scores = torch.sigmoid(risk_scores)  # 确保0-1
        risk_level = self.risk_classifier(fused_features)
        
        return {
            'emotion_pred': emotion_pred,
            'risk_scores': risk_scores,
            'risk_level': risk_level,
            'fused_features': fused_features
        }

class ModelConfig:
    def __init__(self):
        self.freeze_text_layers = 8 
        self.text_model_name = 'bert-base-chinese'
        self.output_emotion_dim = 3
        self.output_risk_dim = 3
        self.dropout_prob = 0.3