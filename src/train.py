import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import argparse
import gc

# 环境变量优化
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from chsims_dataset import CHSIMSOfflineDataset, SimpleBatchProcessor
    from multimodal_student import StudentMultimodalModel, ModelConfig
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('--pt_root', type=str, default='/root/autodl-tmp/SIMS_PT', help='预处理后的.pt文件夹根目录')
parser.add_argument('--bert_path', type=str, default='./bert-base-chinese', help='BERT路径')
args = parser.parse_args()
BERT_MODEL_PATH = args.bert_path

# ==========================================
# 🔥 核心方案二：定义 Focal Loss 分类神器
# ==========================================
class FocalLoss(nn.Module):
    """
    针对类别严重不平衡的 Focal Loss。
    数学原理: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight # 这里的 weight 就是 alpha，用来平衡类别数量差异
        self.gamma = gamma   # gamma 用来降低易分样本的权重，通常设为 2.0
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. 计算自带 weight 的标准交叉熵 (不求平均，保留每个样本的 loss)
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        # 2. 逆向推导出模型对真实标签的预测概率 p_t
        pt = torch.exp(-ce_loss)
        # 3. 加上 Focal 机制的惩罚项: (1 - p_t)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_one_epoch(model, dataloader, optimizer, device, epoch, writer, scaler, focal_criterion):
    model.train()
    total_loss_accum = 0
    num_batches = len(dataloader)
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(dataloader):
        if not batch or len(batch) == 0: continue
            
        try:
            text_ids = batch['text_input_ids'].to(device, non_blocking=True)
            attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            video_frames = batch['video_frames'].to(device, non_blocking=True)
            audio_mfcc = batch['audio_mfcc'].to(device, non_blocking=True)
            audio_mel = batch['audio_mel'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
        except KeyError: continue
        
        with torch.cuda.amp.autocast():
            outputs = model(text_ids, attention_mask, video_frames, audio_mfcc, audio_mel)
            loss_emotion = nn.functional.mse_loss(outputs['emotion_pred'], targets)
            
            # 伪风险计算
            pseudo_risk = torch.where(targets < 0, 
                          torch.abs(targets) * 0.5 + 0.2, # 负向情绪：风险随烈度从 0.2 飙升
                          torch.tensor(0.1).to(targets.device)) # 正向/平静：固定为低风险
            pseudo_risk = torch.clamp(pseudo_risk, min=0.0, max=1.0)
            loss_risk = nn.functional.mse_loss(outputs['risk_scores'], pseudo_risk)
            
            risk_mean = pseudo_risk.mean(dim=1)
            pseudo_level = torch.zeros_like(risk_mean, dtype=torch.long)
            pseudo_level[risk_mean > 0.55] = 2  # 只有真正的深度负面情绪才会超过 0.5
            pseudo_level[(risk_mean > 0.15) & (risk_mean <= 0.55)] = 1
            
            # 🔥 采用 Focal Loss 替代原始的 CrossEntropy
            loss_level = focal_criterion(outputs['risk_level'], pseudo_level)
            
            # 综合多任务 Loss
            loss = 1.0 * loss_emotion + 0.5 * loss_risk + 0.3 * loss_level
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss_accum += loss.item()
        
        if batch_idx % 5 == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
            print(f'Epoch {epoch+1} [{batch_idx}/{num_batches}] | Loss: {loss.item():.4f}')
        del text_ids, attention_mask, video_frames, audio_mfcc, audio_mel, targets, outputs, loss
        # 每 10 个 batch 强行呼叫一次底层垃圾回收车
        if batch_idx % 10 == 0:
            gc.collect()
    return total_loss_accum / num_batches

def validate_model(model, dataloader, device, focal_criterion):
    model.eval()
    val_loss = 0
    num_batches = 0
    with torch.inference_mode(): 
        for batch in dataloader:
            if not batch or len(batch) == 0: continue
            text_ids = batch['text_input_ids'].to(device, non_blocking=True)
            attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            video_frames = batch['video_frames'].to(device, non_blocking=True)
            audio_mfcc = batch['audio_mfcc'].to(device, non_blocking=True)
            audio_mel = batch['audio_mel'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(text_ids, attention_mask, video_frames, audio_mfcc, audio_mel)
                loss_emotion = nn.functional.mse_loss(outputs['emotion_pred'], targets)
                
                pseudo_risk = torch.where(targets < 0, 
                          torch.abs(targets) * 0.5 + 0.2, # 负向情绪：风险随烈度从 0.2 飙升
                          torch.tensor(0.1).to(targets.device)) # 正向/平静：固定为低风险
                pseudo_risk = torch.clamp(pseudo_risk, min=0.0, max=1.0)
                loss_risk = nn.functional.mse_loss(outputs['risk_scores'], pseudo_risk)
                
                risk_mean = pseudo_risk.mean(dim=1)
                pseudo_level = torch.zeros_like(risk_mean, dtype=torch.long)
                pseudo_level[risk_mean > 0.55] = 2  # 只有真正的深度负面情绪才会超过 0.5
                pseudo_level[(risk_mean > 0.15) & (risk_mean <= 0.55)] = 1
                
                # 🔥 验证集同样使用 Focal Loss 评判
                loss_level = focal_criterion(outputs['risk_level'], pseudo_level)
                
                total_loss = 1.0 * loss_emotion + 0.5 * loss_risk + 0.3 * loss_level
            
            val_loss += total_loss.item()
            num_batches += 1
            del text_ids, attention_mask, video_frames, audio_mfcc, audio_mel, targets, outputs, total_loss
            if num_batches % 10 == 0:
                gc.collect()
    return val_loss / num_batches

def main():
    def fix_seed(seed=42):
        import torch, numpy as np, random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True # 开启加速
        torch.backends.cuda.matmul.allow_tf32 = True # 开启 TF32 (3090 核心优化)
        torch.backends.cudnn.allow_tf32 = True
    fix_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)
    writer = SummaryWriter(log_dir='runs/3090_final_optimized_v3')
    
    print("\n加载离线数据集...")
    train_dir, val_dir = os.path.join(args.pt_root, 'train_pt'), os.path.join(args.pt_root, 'valid_pt')
    train_dataset, val_dataset = CHSIMSOfflineDataset(train_dir), CHSIMSOfflineDataset(val_dir)
    
    # 保持高效加载
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=SimpleBatchProcessor(), num_workers=0, 
                              pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=SimpleBatchProcessor(), num_workers=0, 
                            pin_memory=False)
    
    print("\n创建模型并配置进阶策略...")
    config = ModelConfig()
    model = StudentMultimodalModel(config=config, bert_model_path=BERT_MODEL_PATH).to(device)
    
    # ==========================================
    # 🔥 核心方案三：解除 BERT 封印 (改为仅冻结前 4 层)
    # ==========================================
    # 兼容不同的属性命名
    bert_module = getattr(model, 'text_encoder', getattr(model, 'bert', None))
    if bert_module is not None:
        for layer in bert_module.encoder.layer[:4]:
            for param in layer.parameters():
                param.requires_grad = False
        print(">>> 已成功解除部分封印：仅冻结 BERT 前 4 层参数，释放更强语义能力。")
    
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                            lr=5e-5, weight_decay=0.05)
    
    # 加入学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # ==========================================
    # 🔥 核心方案一 & 二混合：实例化带 Alpha 权重的 Focal Loss
    # ==========================================
    # Alpha 权重调整为 [1.5, 1.0, 2.5]，搭配 Focal Loss 自动降权的 gamma=2.0
    class_weights = torch.tensor([1.5, 1.0, 2.0]).to(device)
    focal_criterion = FocalLoss(weight=class_weights, gamma=2.0).to(device)
    print(">>> 已激活 Focal Loss 分类器并挂载最佳惩罚权重 [1.5, 1.0, 2.0]。")

    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/20")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, scaler, focal_criterion)
        val_loss = validate_model(model, val_loader, device, focal_criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        writer.add_scalars('Loss/Summary', {'Train': train_loss, 'Val': val_loss}, epoch)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, 'checkpoints/best_model.pth')
            print(f"*** 发现更优模型 (Val Loss: {val_loss:.4f})，已更新 best_model.pth")
        gc.collect()
        torch.cuda.empty_cache()
    writer.close()

if __name__ == "__main__":
    main()