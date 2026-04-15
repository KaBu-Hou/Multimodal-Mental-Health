# bert_local.py
import os
import sys
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

def get_local_bert_path():
    """获取本地BERT模型路径"""
    possible_paths = [
        './bert-base-chinese',
        '../bert-base-chinese',
        './bert-base-chinese',
        '../bert-base-chinese',
        os.path.expanduser('~/.cache/huggingface/hub/models--bert-base-chinese'),
        os.path.expanduser('~/.cache/huggingface/hub/models--bert-base-chinese/snapshots'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # 如果是快照目录，取最新版本
            if 'snapshots' in path and os.path.isdir(path):
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    latest = max(subdirs)
                    return os.path.join(path, latest)
            return path
    
    return None

def load_bert_local():
    """从本地加载BERT模型和分词器"""
    local_path = get_local_bert_path()
    
    if local_path and os.path.exists(local_path):
        print(f"从本地加载BERT模型: {local_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            model = AutoModel.from_pretrained(local_path, local_files_only=True)
            return tokenizer, model
        except Exception as e:
            print(f"本地加载失败: {e}")
    
    # 如果本地没有，尝试使用备用方法
    print("本地BERT模型未找到，使用备用方案...")
    
    # 方法1: 使用较小的中文模型
    try:
        # 尝试清华的预训练模型
        print("尝试使用清华预训练模型...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', 
                                                  cache_dir='./cache',
                                                  force_download=False,
                                                  local_files_only=True)
        model = BertModel.from_pretrained('bert-base-chinese',
                                          cache_dir='./cache',
                                          force_download=False,
                                          local_files_only=True)
        return tokenizer, model
    except Exception as e:
        print(f"加载失败: {e}")
    
    # 方法2: 使用简单的embedding层
    print("使用简单的embedding层作为替代...")
    from transformers import BertConfig
    
    # 创建一个最小的BERT配置
    config = BertConfig(
        vocab_size=21128,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
    )
    
    model = BertModel(config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', 
                                              do_lower_case=False)
    return tokenizer, model
