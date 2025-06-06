import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import argparse
import logging
import time
from tqdm import tqdm
from models import SetTransformer
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run_name', type=str, default='TAPES_experiment')
parser.add_argument('--num_steps', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=400)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型初始化 (D为特征数)
data_path = 'data/raw/TAPES_Data_Harvard_random50.csv'
data_df = pd.read_csv(data_path)

# 选择仅数值型列（自动筛选）
numeric_data_df = data_df.select_dtypes(include=[np.number])

# 检查筛选后的列
print(f"Numeric columns used: {numeric_data_df.columns.tolist()}")

# 转换数据为float32
X_np = numeric_data_df.values.astype(np.float32)

# PCA降维到512维
# pca = PCA(n_components=512)
# PCA降维到最大允许值 (50)
pca = PCA(n_components=50)
X_np_reduced = pca.fit_transform(X_np)

# 转换数据为Tensor
# 关键改动，加上unsqueeze(0)适应模型
X_tensor = torch.from_numpy(X_np_reduced).unsqueeze(0).to(device)
# 根据数据维度自动定义模型维度
# 明确维度
batch_size, set_size, feature_dim = X_tensor.shape
print(f"Batch size: {batch_size}, Set size: {set_size}, Feature dim: {feature_dim}")

D = feature_dim  # 明确为feature_dim
K = 5  # Attention heads数目，可以根据需要修改
dim_output = D  # 可以根据你的实验调整

net = SetTransformer(D, K, dim_output).to(device)

# 模型保存路径
save_dir = os.path.join('experiments', 'set_transformer', args.run_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# 定义训练函数
def train():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(logging.FileHandler(
        os.path.join(save_dir, 'train_' + time.strftime('%Y%m%d-%H%M') + '.log'),
        mode='w'))
    logger.info(str(args) + '\n')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for t in tqdm(range(1, args.num_steps + 1)):
        net.train()
        optimizer.zero_grad()

        # 训练：简单示例 (自监督示例，可调整)
        pred = net(X_tensor)
        loss = criterion(pred, X_tensor)  # 此处为示例，具体损失函数需根据实际任务调整
        loss.backward()
        optimizer.step()

        if t % args.test_freq == 0:
            logger.info(f'Step {t}, Loss {loss.item()}')

        if t % args.save_freq == 0:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join(save_dir, 'model.tar'))

    torch.save({'state_dict': net.state_dict()},
               os.path.join(save_dir, 'model.tar'))

if __name__ == '__main__':
    if args.mode == 'train':
        train()

