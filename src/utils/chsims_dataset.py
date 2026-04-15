import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import librosa
from PIL import Image
import torchvision.transforms as transforms
import warnings
from transformers import AutoTokenizer
warnings.filterwarnings('ignore')

# 导入本地BERT加载器
from bert_local import load_bert_local

class CHSIMSDataset(Dataset):
    """CH-SIMS多模态情感数据集处理类"""
    def __init__(self, data_dir='../../SIMS', split='train', transform=None, 
                 target_type='dimensional', max_text_length=128, 
                 bert_model_path='bert-base-chinese'):
        """
        参数:
            bert_model_path: 本地BERT模型路径
        """
        self.failed_videos = set()
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'Raw')
        self.split = split
        self.target_type = target_type
        self.max_text_length = max_text_length
        
        # 加载标签
        labels_path = os.path.join(data_dir, 'label.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"标签文件不存在: {labels_path}")
        self.df = pd.read_csv(labels_path)
        
        # 根据split筛选数据
        if 'split' in self.df.columns:
            split_map = {'train': 0, 'valid': 1, 'test': 2}
            if split in split_map:
                self.df = self.df[self.df['split'] == split_map[split]]
        elif 'mode' in self.df.columns:
            self.df = self.df[self.df['mode'] == split]
        
        self.df = self.df.reset_index(drop=True)
        
        # 使用本地BERT分词器（调用bert_local.py）
        print(f"正在加载本地BERT模型和分词器...")
        try:
            self.tokenizer, _ = load_bert_local()
            if self.tokenizer:
                print("✓ 本地BERT分词器加载成功")
            else:
                raise ValueError("load_bert_local()返回None")
        except Exception as e:
            import traceback
            print(f"加载本地BERT分词器失败: {e}")
            print(f"详细错误：\n{traceback.format_exc()}")
            print("使用备用方案：从Hugging Face加载")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        
        # 图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # 音频参数
        self.sr = 16000
        self.n_mfcc = 13
        
        # 缓存
        self.cache = {}
        
        print(f"数据集初始化完成: {split} 划分, 样本数: {len(self)}")
        
        # ========================
        # 新增：自动验证有效样本（关键！）
        # ========================
        self._validate_data_quality()
    
    def _validate_data_quality(self):
        """验证数据集质量，统计有效样本比例"""
        print("\n" + "="*60)
        print("正在验证数据集质量...")
        valid_count = 0
        check_num = min(20, len(self)) # 检查前20个样本
        
        for i in range(check_num):
            try:
                sample = self.__getitem__(i)
                if not sample['is_empty']:
                    # 检查特征是否非全零
                    video_ok = not torch.allclose(sample['video_frames'], torch.zeros_like(sample['video_frames']))
                    audio_ok = not torch.allclose(sample['audio_mfcc'], torch.zeros_like(sample['audio_mfcc']))
                    text_ok = len(sample['text'].strip()) > 0
                    
                    if video_ok or audio_ok or text_ok:
                        valid_count += 1
                        status = "✅"
                    else:
                        status = "⚠️ (特征全空)"
                    
                    print(f"  [{i+1}/{check_num}] {status} ID:{sample['video_id']}_{sample['clip_id']} | Text:{sample['text'][:20]}...")
            except Exception as e:
                print(f"  [{i+1}/{check_num}] ❌ 加载失败: {e}")
        
        valid_ratio = valid_count / check_num
        print(f"\n数据质量报告:")
        print(f"  有效样本: {valid_count}/{check_num} ({valid_ratio*100:.1f}%)")
        
        if valid_ratio < 0.5:
            print("\n" + "❌"*20)
            print("严重警告：有效样本比例低于50%！")
            print("请检查：")
            print("1. ffmpeg 是否已安装？")
            print("2. 视频路径是否正确？")
            print("3. label.csv 中的 video_id/clip_id 是否匹配？")
            print("❌"*20 + "\n")
        elif valid_ratio < 0.8:
            print("\n⚠️ 警告：有效样本比例一般，建议检查数据")
        else:
            print("✅ 数据质量良好！")
        print("="*60 + "\n")
    
    def __len__(self):
        return len(self.df)
    
    def extract_video_frames(self, video_path, num_frames=16):
        """从视频中提取帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return torch.zeros((num_frames, 3, 224, 224))
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count <= 0:
                cap.release()
                return torch.zeros((num_frames, 3, 224, 224))
            
            # 确保索引不越界
            if frame_count >= num_frames:
                indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
                indices = np.clip(indices, 0, frame_count-1)
            else:
                indices = np.arange(frame_count)
                indices = np.pad(indices, (0, max(0, num_frames - frame_count)), 
                               mode='edge')
                indices = indices[:num_frames]
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_tensor = self.transform(frame_pil)
                        frames.append(frame_tensor)
                    else:
                        frames.append(torch.zeros(3, 224, 224))
                else:
                    frames.append(torch.zeros(3, 224, 224))
            
            cap.release()
            
            if len(frames) < num_frames:
                frames.extend([torch.zeros(3, 224, 224)] * (num_frames - len(frames)))
            elif len(frames) > num_frames:
                frames = frames[:num_frames]
            
            return torch.stack(frames)
            
        except Exception as e:
            print(f"提取视频帧失败 {video_path}: {e}")
            return torch.zeros((num_frames, 3, 224, 224))
    
    def extract_audio_features_safe(self, video_path):
        """【修复版】先用ffmpeg提取音频，再用librosa处理"""
        try:
            if not os.path.exists(video_path):
                return self.get_empty_audio_features()
            
            import tempfile
            import subprocess
            temp_audio_path = None
            try:
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio_path = temp_audio.name
                temp_audio.close()
                
                # FFmpeg 提取音频
                ffmpeg_cmd = [
                    'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                    '-ar', str(self.sr), '-y', temp_audio_path
                ]
                
                result = subprocess.run(
                    ffmpeg_cmd, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode != 0:
                    return self.get_empty_audio_features()
                
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) < 100:
                    return self.get_empty_audio_features()
                
                audio, sr = librosa.load(temp_audio_path, sr=self.sr)
                if len(audio) == 0:
                    return self.get_empty_audio_features()
                
                # 提取特征
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 归一化
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
                mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
                
                def adjust_time_dim(feature, target_length=100):
                    if feature.shape[1] >= target_length:
                        return feature[:, :target_length]
                    else:
                        return np.pad(feature, ((0, 0), (0, target_length - feature.shape[1])), mode='constant')
                
                return {
                    'mfcc': adjust_time_dim(mfcc).T,
                    'mel_spec': adjust_time_dim(mel_spec_db).T,
                    'chroma': np.zeros((100, 12)), # 简化，暂时不用chroma
                    'pitch': 0.0,
                    'energy': 0.0
                }
                
            except subprocess.TimeoutExpired:
                return self.get_empty_audio_features()
            finally:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try: os.unlink(temp_audio_path)
                    except: pass
            
        except Exception as e:
            return self.get_empty_audio_features()
    
    def get_empty_audio_features(self):
        """获取空的音频特征"""
        return {
            'mfcc': np.zeros((100, self.n_mfcc)),
            'mel_spec': np.zeros((100, 128)),
            'chroma': np.zeros((100, 12)),
            'pitch': 0.0,
            'energy': 0.0
        }
    
    def get_target(self, row):
        """获取目标标签"""
        if self.target_type == 'dimensional':
            try:
                target = torch.tensor([
                    float(row['label_T']),
                    float(row['label_A']), 
                    float(row['label_V'])
                ], dtype=torch.float32)
            except:
                target = torch.zeros(3, dtype=torch.float32)
        else:
            target = torch.zeros(3, dtype=torch.float32)
        
        return target
    
    def create_empty_sample(self, row):
        """创建空样本"""
        text_encoding = self.tokenizer(
            "", max_length=self.max_text_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        try:
            target = self.get_target(row)
        except:
            target = torch.zeros(3, dtype=torch.float32)
        
        return {
            'video_frames': torch.zeros((16, 3, 224, 224)),
            'audio_mfcc': torch.zeros((100, self.n_mfcc)),
            'audio_mel': torch.zeros((100, 128)),
            'audio_chroma': torch.zeros((100, 12)),
            'audio_pitch': torch.tensor(0.0),
            'audio_energy': torch.tensor(0.0),
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'target': target,
            'video_id': row.get('video_id', 'unknown'),
            'clip_id': str(row.get('clip_id', '0000')),
            'text': str(row.get('text', '')),
            'mode': row.get('mode', 'unknown'),
            'is_empty': True
        }
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        row = self.df.iloc[idx]
        
        # 构建视频路径
        video_id = row['video_id']
        clip_id = str(row['clip_id']).zfill(4)
        video_path = os.path.join(self.raw_dir, video_id, f"{clip_id}.mp4")
        
        if video_path in self.failed_videos or not os.path.exists(video_path):
            self.failed_videos.add(video_path)
            sample = self.create_empty_sample(row)
            self.cache[idx] = sample
            return sample
        
        try:
            video_frames = self.extract_video_frames(video_path)
            audio_features = self.extract_audio_features_safe(video_path)
            text = str(row['text'])
            
            text_encoding = self.tokenizer(
                text, max_length=self.max_text_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            
            target = self.get_target(row)
            
            sample = {
                'video_frames': video_frames,
                'audio_mfcc': torch.tensor(audio_features['mfcc'], dtype=torch.float32),
                'audio_mel': torch.tensor(audio_features['mel_spec'], dtype=torch.float32),
                'audio_chroma': torch.tensor(audio_features['chroma'], dtype=torch.float32),
                'audio_pitch': torch.tensor(audio_features['pitch'], dtype=torch.float32),
                'audio_energy': torch.tensor(audio_features['energy'], dtype=torch.float32),
                'text_input_ids': text_encoding['input_ids'].squeeze(0),
                'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
                'target': target,
                'video_id': video_id,
                'clip_id': clip_id,
                'text': text,
                'mode': row.get('mode', 'unknown'),
                'is_empty': False
            }
            
            self.cache[idx] = sample
            return sample
            
        except Exception as e:
            self.failed_videos.add(video_path)
            sample = self.create_empty_sample(row)
            self.cache[idx] = sample
            return sample

class SimpleBatchProcessor:
    """简化的批处理器"""
    def __call__(self, batch):
        batch_dict = {}
        batch = [s for s in batch if s is not None and not s.get('is_empty', False)]
        if not batch: return {}
        
        for key in batch[0].keys():
            if key in ['video_id', 'clip_id', 'text', 'mode' , 'is_empty']:
                batch_dict[key] = [s[key] for s in batch]
            else:
                try:
                    batch_dict[key] = torch.stack([s[key] for s in batch])
                except:
                    batch_dict[key] = [s[key] for s in batch]
        return batch_dict

# ========================
# 直接运行此文件可测试数据集
# ========================
if __name__ == "__main__":
    print("="*60)
    print("测试 CHSIMSDataset")
    print("="*60)
    
    # 注意：请修改下面的 data_root 为你实际的数据集路径
    try:
        dataset = CHSIMSDataset(
            data_dir='../../SIMS', # 改成你的路径
            split='train'
        )
        
        if len(dataset) > 0:
            print("\n读取第一个样本测试...")
            sample = dataset[0]
            print("样本结构:")
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}")
                elif k in ['text', 'video_id']:
                    print(f"  {k}: {v}")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
import glob

# 这是专门为离线 .pt 文件准备的极速加载器
class CHSIMSOfflineDataset(Dataset):
    def __init__(self, pt_dir):
        # 获取目录下所有的 .pt 文件路径
        self.pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
        if len(self.pt_files) == 0:
            raise ValueError(f"在 {pt_dir} 中没有找到任何 .pt 文件，请先运行预处理脚本！")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        # 瞬间读取预处理好的张量，0 CPU 负担！
        return torch.load(self.pt_files[idx])