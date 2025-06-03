import argparse
from utils import *
from torch.utils.tensorboard import SummaryWriter

def train_cifar10(config):
    from dataset import get_cifar10
    train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_labeled_loader, train_unlabeled_loader, test_loader = get_cifar10(config, config.data_root)
    device = get_device(config.device)
    tensorboard_writer = SummaryWriter(config.tensorboard_dir)
    
    from trainer import FixMatchTrainer
    trainer = FixMatchTrainer(config, device, tensorboard_writer, train_labeled_dataset, train_unlabeled_dataset, test_dataset)
    trainer.train(train_labeled_loader, train_unlabeled_loader, test_loader)
    
    tensorboard_writer.close()

def update_config(config):
    if config.dataset == 'cifar10':
        config.num_classes = 10
        if config.arch == 'wideresnet':
            config.model_depth = 28
            config.model_width = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("--workspace_name", type=str, help="sub directory name", required=True)
    args = parser.parse_args()
    # load config from yaml file
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_path)
    # 保存配置文件路径
    config.config_path = args.config_path
    config.workspace_name = args.workspace_name

    return config

if __name__ == "__main__":
    config = parse_args()
    config = prepare_workspace(config)
    config_file_name = os.path.basename(config.config_path)
    init_seed(config.seed)
    update_config(config)
    
    if config.task == 'cifar10':
        train_cifar10(config)
    else:
        raise ValueError(f"Unknown config file: {config_file_name}")