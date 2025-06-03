import torch
import numpy as np
import random
import os
from omegaconf import OmegaConf
from utils.save_git_info import save_git_info

def get_device(device:str):
    """
    获取设备
    
    Args:
        device (str): 设备
        
    Returns:
        torch.device: 设备
    """
    if torch.cuda.is_available() and device[:5] == "cuda:":
        return torch.device(device)
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device: {device}")
    
def init_seed(seed):
    """
    初始化所有随机数种子，确保实验可重复性
    
    Args:
        seed (int): 随机数种子
    """
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # NumPy (同时影响scipy.stats的随机性)
    np.random.seed(seed)
    
    # Python random
    random.seed(seed)
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_workspace(config:OmegaConf):
    """
    准备工作空间，创建必要的目录结构并保存配置文件、git信息
    
    Args:
        config: 配置对象，用于存储工作空间相关路径
        
    Returns:
        config: 更新后的配置对象，包含以下新增属性:
            - workspace (str): 工作空间根目录路径
            - model_dir (str): 模型保存目录路径
            - tensorboard_dir (str): TensorBoard日志目录路径
            - log_dir (str): 日志文件目录路径
    """
    # 获取配置文件名
    config_filename = os.path.basename(config.config_path)
    
    # 创建工作空间目录结构
    config.workspace = os.path.join('workspace', config.workspace_name)
    if not os.path.exists(config.workspace):
        os.makedirs(config.workspace)
        
    # 设置各个子目录路径
    config.model_dir = os.path.join(config.workspace, 'model')
    config.tensorboard_dir = os.path.join(config.workspace, 'tensorboard')
    config.log_dir = os.path.join(config.workspace, 'log')
    
    # 创建必要的子目录
    for directory in [config.model_dir, config.tensorboard_dir, config.log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 保存配置文件
    yaml_conf = OmegaConf.to_yaml(config)
    config_save_path = os.path.join(config.log_dir, config_filename)
    with open(config_save_path, 'w') as f:
        f.write(yaml_conf)
    print(f"save config to: {config_save_path}")
    
    save_git_info(config.log_dir)
    
    return config