多模态大学生健康心理风险检测系统(Multimodal-Mental-Health-Detection)

项目简介
本项目是一个基于学生自主记录视频的心理健康检测系统。通过融合**文本（风险语义）、音频（声学特征）和视觉（面部表情）**三个维度的信息，系统能够自动评估学生当前的心理风险等级（低、中、高）以及情感连续分数。
核心特色引入了Focal Loss解决类别不平衡问题，并通过部分解冻BERT层显着提升了图像理解深度，能够捕捉到的情绪波动。

核心技术栈
文本模态：预训练 BERT (Chinese-Base) + 语义解封策略
音频模态: 实体 LSTM (Bi-LSTM) 捕捉语调变化
园林模态：3D-CNN 提取了一只可爱的动态特征
融合机制：特征融合级（Feature-level Fusion）+全连接决策层
损失函数：带有动态惩罚权重的焦点损失

📂 项目结构
├── checkpoints/          # 存放训练好的 best_model.pth 权重
├── data/                 # 数据集读取说明与预处理逻辑
├── src/                  # 核心源代码
│   ├── models/           # 模型定义 (multimodal_student.py)
│   ├── utils/            # 数据集加载器 (chsims_dataset.py)
│   ├── train.py          # 训练脚本 (带自动调参逻辑)
│   └── test.py           # 离线评估脚本
├── app.py                # 基于 Gradio 的可视化演示交互界面
├── requirements.txt      # 环境依赖清单
└── README.md             # 项目说明文档

⚙️环境安装
建议在Linux服务器（如AutoDL）上运行，推荐CUDA 11.8+环境。
# 克隆仓库
git clone https://github.com/YourUsername/Multimodal-Mental-Health.git
cd Multimodal-Mental-Health
# 安装依赖
pip install -r requirements.txt

📊 训练与评估
1. 资料准备
本项目使用CH-SIMS数据集。请确保将视频数据制作为.pt离线张量包，路径指向pt_root。

2.开始训练
支持自动降低学习率（ReduceLROnPlateau）和TensorBoard监控
python src/train.py --pt_root /path/to/your/data --bert_path ./bert-base-chinese

3. 模型评估
运行测试脚本查看各风险等级的 Precision/Recall 指标：
python src/test.py

🌐 Gradio交互展示
项目集成了基于Gradio的Web界面，支持实时上传视频并获取诊断报告。
python app.py
功能特性：
实时风险推理：秒级输出心理等级。
托盘展示：同步输出情感连续分数的MSE 托盘。
定向预警：针对高风险信号（如低沉语调+悲伤词汇）进行红色预警。

📈 实验结果
指标项,                表现,          备注
高风险召回率（Recall）,  68%,          针对严重心理问题的识别能力极强
总体比重F1,             31.6%,       基于CH-SIMS 三分类任务
情感回归 (MSE),         0.33,         连续情感得分预测平滑度较低

🤝 贡献与致谢
感谢CH-SIMS团队提供的中文多模情感数据集。
参考了BERT和相关多模态融合论文的先进思路。
