import os
import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

# 导入你原来的数据集类
from chsims_dataset import CHSIMSDataset 

# 🚨 修复关键点：将 lambda 函数提取到全局作用域，写成标准函数
def custom_collate_fn(batch):
    return batch[0]

def preprocess_and_save(data_root, bert_path, save_root, num_workers=10):
    # 创建保存预处理数据的文件夹
    valid_save_dir = os.path.join(save_root, 'valid_pt')
    os.makedirs(valid_save_dir, exist_ok=True)

    splits = [('valid', valid_save_dir)]
    
    for split, save_dir in splits:
        print(f"\n🚀 开始高速预处理 {split} 集数据...")
        # 实例化你原来的数据集
        dataset = CHSIMSDataset(data_dir=data_root, split=split, bert_model_path=bert_path)
        
        # 核心提速：利用 14 核 CPU 开启多进程并行
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=custom_collate_fn, # 👈 修改这里：调用外部的正式函数
            prefetch_factor=2,
            pin_memory=False
        )
        
        valid_count = 0
        # 遍历 dataloader
        for sample in tqdm(dataloader, desc=f"Processing {split} (Workers: {num_workers})"):
            if sample is None or sample.get('is_empty', False):
                continue
                
            file_path = os.path.join(save_dir, f"sample_{valid_count:05d}.pt")
            torch.save(sample, file_path)
            valid_count += 1
            
        print(f"✅ {split} 集预处理完成！共保存 {valid_count} 个有效样本至 {save_dir}")

if __name__ == "__main__":
    # 针对 RTX 3090 的多进程安全启动模式
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("警告: 无法设置 multiprocessing 启动方法为 spawn。")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/SIMS', help='原始数据集路径')
    parser.add_argument('--bert_path', type=str, default='./bert-base-chinese', help='BERT路径')
    parser.add_argument('--save_root', type=str, default='/root/autodl-tmp/SIMS_PT', help='保存.pt文件的路径')
    parser.add_argument('--num_workers', type=int, default=10, help='并行处理的进程数 (建议 8-12)') 
    args = parser.parse_args()
    
    preprocess_and_save(args.data_root, args.bert_path, args.save_root, args.num_workers)