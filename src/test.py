import torch
import os
import sys
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 导入你的模块
try:
    from chsims_dataset import CHSIMSOfflineDataset, SimpleBatchProcessor
    from multimodal_student import StudentMultimodalModel, ModelConfig
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--pt_root', type=str, default='/root/autodl-tmp/SIMS_PT', help='预处理文件夹')
parser.add_argument('--bert_path', type=str, default='./bert-base-chinese', help='BERT路径')
parser.add_argument('--ckpt_path', type=str, default='checkpoints/best_model.pth', help='模型权重路径')
args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"🎓 心理健康多模态检测：终极评估报告 🎓")
    print(f"{'='*50}")
    print(f"使用设备: {device}")

    # 1. 检查测试集
    test_dir = os.path.join(args.pt_root, 'test_pt')
    if not os.path.exists(test_dir):
        print(f"\n❌ 致命错误: 找不到测试集目录 {test_dir}")
        print("请先修改 preprocess_dataset.py 提取 'test' 集数据！")
        return

    print("\n[1/3] 正在加载测试集数据...")
    test_dataset = CHSIMSOfflineDataset(test_dir)
    # 测试时不需要打乱，且 num_workers 开 4 个就足够快了
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                             collate_fn=SimpleBatchProcessor(), num_workers=0, 
                             pin_memory=True, persistent_workers=False)
    print(f"✓ 测试集样本总数: {len(test_dataset)}")

    print("\n[2/3] 正在初始化模型并加载最佳权重...")
    config = ModelConfig()
    model = StudentMultimodalModel(config=config, bert_model_path=args.bert_path).to(device)

    # 加载 best_model.pth
    if not os.path.exists(args.ckpt_path):
        print(f"\n❌ 找不到权重文件: {args.ckpt_path}")
        return
        
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    # 兼容处理：检查存的是裸字典还是包裹的字典
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"✓ 成功挂载最强权重: {args.ckpt_path}")

    model.eval()

    # 存储预测结果和真实标签
    all_true_levels = []
    all_pred_levels = []
    all_true_emotions = []
    all_pred_emotions = []

    print("\n[3/3] 🚀 开始推理预测 (Inference)...")
    with torch.inference_mode():
        for batch in test_loader:
            if not batch or len(batch) == 0: continue
            text_ids = batch['text_input_ids'].to(device, non_blocking=True)
            attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            video_frames = batch['video_frames'].to(device, non_blocking=True)
            audio_mfcc = batch['audio_mfcc'].to(device, non_blocking=True)
            audio_mel = batch['audio_mel'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(text_ids, attention_mask, video_frames, audio_mfcc, audio_mel)

            # ====== 提取分类任务（风险等级）数据 ======
            # 1. 根据你的训练逻辑，还原出测试集的真实风险等级 (Ground Truth)
            pseudo_risk = torch.abs(targets) * 0.4 + 0.3
            pseudo_risk = torch.clamp(pseudo_risk, min=0.0, max=1.0)
            risk_mean = pseudo_risk.mean(dim=1)
            true_level = torch.zeros_like(risk_mean, dtype=torch.long)
            true_level[risk_mean > 0.6] = 2
            true_level[(risk_mean > 0.4) & (risk_mean <= 0.6)] = 1

            # 2. 获取模型预测的类别 (概率最大的那一类)
            pred_level = outputs['risk_level'].argmax(dim=1)

            all_true_levels.extend(true_level.cpu().numpy())
            all_pred_levels.extend(pred_level.cpu().numpy())

            # ====== 提取回归任务（情感预测）数据 ======
            all_true_emotions.append(targets.cpu().numpy())
            all_pred_emotions.append(outputs['emotion_pred'].cpu().numpy())

    # ================= 成绩结算与报告 =================
    print("\n" + "🌟"*25)
    print("      🎯 心理风险等级预测 (分类) 成绩单")
    print("🌟"*25)
    
    # 因为有可能有些批次里没有某些类别，我们动态获取存在的类别名称
    target_names = ['低风险 (Level 0)', '中风险 (Level 1)', '高风险 (Level 2)']
    unique_labels = np.unique(all_true_levels)
    actual_target_names = [target_names[i] for i in unique_labels]

    acc = accuracy_score(all_true_levels, all_pred_levels)
    f1 = f1_score(all_true_levels, all_pred_levels, average='macro')

    print(f"👉 总体准确率 (Accuracy): {acc * 100:.2f}%")
    print(f"👉 总体综合F1 (Macro F1): {f1 * 100:.2f}%\n")

    print("📊 各风险等级详细指标 (Precision/Recall):")
    print(classification_report(all_true_levels, all_pred_levels, 
                                labels=unique_labels,
                                target_names=actual_target_names, 
                                zero_division=0))

    print("\n" + "-"*50)
    print("📈 情感连续分数预测 (回归) 误差情况")
    print("-"*50)
    true_emotions_np = np.vstack(all_true_emotions)
    pred_emotions_np = np.vstack(all_pred_emotions)
    mse = mean_squared_error(true_emotions_np, pred_emotions_np)
    print(f"👉 总体均方误差 (MSE): {mse:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()