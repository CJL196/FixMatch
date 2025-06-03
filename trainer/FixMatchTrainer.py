import torch
from torch import nn
from loguru import logger
import sys
from contextlib import nullcontext
import numpy as np
from scipy import stats
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class FixMatchTrainer:
    def __init__(self, config, device, tensorboard_writer, train_labeled_dataset, train_unlabeled_dataset, test_dataset):
        self.initialize(config, device, tensorboard_writer, train_labeled_dataset, train_unlabeled_dataset, test_dataset)
        
        # 初始化模型
        from model import build_wideresnet
        model = build_wideresnet(config.model_depth, config.model_width, config.dropout, config.num_classes)
        
        if config.use_ema:
            from model.ema import ModelEMA
            self.ema = ModelEMA(device, model, config.ema_decay)
        
        self.initialize_model(model)
        
        # 初始化优化器
        self.initialize_optimizer()
        
        # 初始化学习率调度器
        self.initialize_scheduler()
        
        # 设置混合精度训练和日志
        self.setup()

    def initialize(self, config, device, tensorboard_writer, train_labeled_dataset, train_unlabeled_dataset, test_dataset):
        self.config = config
        self.device = device
        self.writer = tensorboard_writer
        self.train_labeled_dataset = train_labeled_dataset
        self.train_unlabeled_dataset = train_unlabeled_dataset
        self.test_dataset = test_dataset
        
        # 基本属性初始化
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.start_epoch = 0
        
        # 用于存储训练和评估指标的字典
        self.metrics = {}
    
    def initialize_model(self, model):
        """初始化模型"""
        self.model = model.to(self.device)
        
        # 加载预训练模型
        if self.config.model_ckpt is not None:
            print(f'load pretrained model from {self.config.model_ckpt}')
            checkpoint = torch.load(self.config.model_ckpt, map_location=self.device)
            if self.config.use_ema:
                self.ema.ema.load_state_dict(checkpoint['ema'])
            
            self.model.load_state_dict(checkpoint['model'], strict=False)
    
    def initialize_optimizer(self, lr=None, weight_decay=None):
        """初始化优化器"""
        if lr is None:
            lr = self.config.learning_rate
        if weight_decay is None:
            weight_decay = self.config.weight_decay
            
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.SGD(grouped_parameters, lr=lr,
                            momentum=0.9, nesterov=True)
        
        # 加载优化器状态
        if self.config.optim_ckpt is not None:
            print(f'load pretrained optimizer from {self.config.optim_ckpt}')
            checkpoint = torch.load(self.config.optim_ckpt, map_location=self.device)
            # 检查是否包含优化器状态
            if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # 恢复训练状态
                if 'global_step' in checkpoint:
                    self.global_step = checkpoint['global_step']
                    print(f'恢复全局步数: {self.global_step}')
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1  # +1表示从下一个epoch开始
                    print(f'从epoch {self.start_epoch}继续训练')
            else:
                # 兼容旧版本的checkpoint格式
                self.optimizer.load_state_dict(checkpoint)
    
    def initialize_scheduler(self, warmup_steps=None, min_lr=None):
        """初始化学习率调度器"""
        self.warmup_steps = warmup_steps if warmup_steps is not None else self.config.warmup_steps
        min_lr = min_lr if min_lr is not None else self.config.min_learning_rate
        
        # 计算总步数用于余弦衰减
        total_steps = self.config.num_epochs * (len(self.train_labeled_dataset) // self.config.batch_size)
        # 创建余弦退火调度器，T_max设置为总步数减去预热步数
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - self.warmup_steps, eta_min=min_lr)
    
    def initialize_amp(self):
        """初始化混合精度训练"""
        # 使用混合精度训练
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        print(f'dtype={dtype}')
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        
        # 根据 PyTorch 版本选择使用 torch.amp 或 torch.cuda.amp
        if hasattr(torch, 'amp'):
            self.ctx = nullcontext() if self.device == torch.device('cpu') else torch.amp.autocast(device_type=self.device.type, dtype=self.ptdtype)
            self.scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
        else:
            self.ctx = nullcontext() if self.device == torch.device('cpu') else torch.cuda.amp.autocast(dtype=self.ptdtype)
            self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    def initialize_logger(self):
        """初始化日志记录器"""
        logger.add(self.config.log_dir + "/logs_{time}.log")
        # logger.add(sys.stdout, colorize=True, format="{message}")
    
    def setup(self):
        """完成所有初始化工作，需要在子类中调用"""
        self.initialize_amp()
        self.initialize_logger()
    
    def adjust_learning_rate(self):
        """根据步数调整学习率，先预热再余弦衰减"""
        if self.global_step < self.warmup_steps:
            # 线性预热
            lr_scale = min(1.0, float(self.global_step) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
        else:
            # 使用预先定义的调度器进行余弦衰减
            self.scheduler.step()
    
    def initialize_metrics(self):
        """初始化指标存储字典"""
        self.metrics = {}
    
    def update_metrics(self, batch_metrics: Dict[str, Any]):
        """
        更新指标存储字典
        batch_metrics: inference函数返回的指标字典
        """
        for key, value in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            # 如果是张量，转换为numpy数组
            if isinstance(value, torch.Tensor):
                value = value.detach()
            
            self.metrics[key].append(value)
    
    def evaluate_metrics_epoch(self, prefix='train', epoch=0):
        """
        评估并记录所有收集的指标
        prefix: 指标前缀（train或test）
        epoch: 当前的训练轮次
        """
        results = {}
        
        for key, values in self.metrics.items():
            if key.startswith('loss'):
                avg_loss = np.mean(values)
                self.writer.add_scalar(f'{prefix}_epoch/{key}', avg_loss, epoch)
                results[key] = avg_loss
            elif key == 'labeled_logits':
                labeled_logits = torch.cat(values, dim=0)
                labeled_label = torch.cat(self.metrics['labeled_label'], dim=0)
                acc1, acc5 = self.accuracy(labeled_logits, labeled_label, topk=(1, 5))
                acc1 = acc1.item()
                acc5 = acc5.item()
                self.writer.add_scalar(f'{prefix}_epoch/acc1', acc1, epoch)
                self.writer.add_scalar(f'{prefix}_epoch/acc5', acc5, epoch)
                results['acc1'] = acc1
                results['acc5'] = acc5
            elif key == 'mask':
                mask = np.mean(values)
                self.writer.add_scalar(f'{prefix}_epoch/mask', mask, epoch)
                results['mask'] = mask
        return results
    
    def evaluate_metrics_step(self, batch_metrics, prefix='train', step=0):
        """
        记录每个step的单值指标到tensorboard
        
        参数:
            batch_metrics: 当前batch的指标字典
            prefix: 指标前缀（train或test）
            step: 当前的训练步数
        """
        # 处理单值指标
        for key, value in batch_metrics.items():
            # 跳过预测值和目标值对
            if key.startswith('loss') and prefix == 'train':
                self.writer.add_scalar(f'{prefix}_step/{key}', value, step)
            elif key == 'mask':
                self.writer.add_scalar(f'{prefix}_step/mask', value, step)
            
        
    
    def save_checkpoint(self, epoch):
        """保存模型和优化器状态"""
        if (epoch + 1) % self.config.save_model_freq == 0:
            if self.config.use_ema:
                torch.save({
                    'model': self.model.state_dict(),
                    'ema': self.ema.ema.state_dict()
                }, self.config.model_dir + "/epoch-" + str(epoch + 1) + ".model")
            else:
                torch.save({'model': self.model.state_dict()}, self.config.model_dir + "/epoch-" + str(epoch + 1) + ".model")
            # 保存优化器状态和训练进度信息
            checkpoint = {
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'epoch': epoch
            }
            torch.save(checkpoint, self.config.model_dir + "/epoch-" + str(epoch + 1) + ".opt")
    
    def inference(self, labeled_batch, unlabeled_batch, model) -> Tuple[torch.Tensor, Dict[str, Any]]:
        metrics = {}
        # labeled
        labeled_img, labeled_label = labeled_batch
        labeled_img = labeled_img.to(self.device)
        labeled_label = labeled_label.to(self.device)
        
        with self.ctx:
            labeled_logits = model(labeled_img)
        Lx = F.cross_entropy(labeled_logits, labeled_label, reduction='mean')
        metrics['loss_Lx'] = Lx.item()
        metrics['labeled_logits'] = labeled_logits.detach().cpu()
        metrics['labeled_label'] = labeled_label.detach().cpu()
        loss = Lx
        
        # unlabeled
        if unlabeled_batch is not None:
            (img_weak, img_strong), _ = unlabeled_batch
            img_weak = img_weak.to(self.device)
            img_strong = img_strong.to(self.device)
            with self.ctx:
                logits_weak = model(img_weak)
                logits_strong = model(img_strong)
            pseudo_label = torch.softmax(logits_weak.detach()/self.config.temperature, dim=-1)
            max_probs, max_idx = pseudo_label.max(dim=-1)
            mask = max_probs.ge(self.config.threshold).float()
            Lu = F.cross_entropy(logits_strong, max_idx, reduction='none')
            Lu = (Lu * mask).mean()
            metrics['loss_Lu'] = Lu.item()
            metrics['mask'] = mask.mean().item()
    
            loss += self.config.lambda_u * Lu
            
        # metrics['loss_total'] = loss.item()
        
        
        return loss, metrics
    
    def backward(self, loss):
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪（如果需要）
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # 更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.epoch = epoch
            # 训练阶段
            self.model.train()
            epoch_loss = 0
            self.initialize_metrics()
            
            unlabeled_iter = iter(train_unlabeled_loader)
            
            for labeled_batch in train_labeled_loader:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(train_unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)
                
                self.optimizer.zero_grad()
                
                # 调整学习率
                self.adjust_learning_rate()
                
                # 推理和计算损失
                loss, batch_metrics = self.inference(labeled_batch, unlabeled_batch, self.model)
                epoch_loss += loss.item()
                
                # 更新指标
                self.update_metrics(batch_metrics)
                self.evaluate_metrics_step(batch_metrics, prefix='train', step=self.global_step)
                
                self.backward(loss)
                
                if self.config.use_ema:
                    self.ema.update(self.model)

                # 更新全局步数
                self.global_step += 1
                
                # 记录当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_rate/lr', current_lr, self.global_step)
            
            # 计算并记录训练指标
            self.evaluate_metrics_epoch(prefix='train', epoch=epoch)
            
            # 记录训练损失
            avg_loss = epoch_loss / len(train_labeled_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            logger.info(f"[Epoch {epoch+1}]: Train loss = {avg_loss:.4f}")
            
            # 保存检查点
            self.save_checkpoint(epoch)
            
            # 测试阶段
            self.model.eval()
            test_loss = 0
            self.initialize_metrics()
            
            if self.config.use_ema:
                test_model = self.ema.ema
            else:
                test_model = self.model
            
            with torch.no_grad():
                for labeled_batch in test_loader:
                    loss, batch_metrics = self.inference(labeled_batch, None, test_model)
                    test_loss += loss.item()
                    self.update_metrics(batch_metrics)
            
            # 计算并记录测试指标
            test_results = self.evaluate_metrics_epoch(prefix='test', epoch=epoch)
            
            # 记录测试损失
            avg_test_loss = test_loss / len(test_loader)
            self.writer.add_scalar('Loss/test', avg_test_loss, epoch)
            
            # 记录测试信息
            logger.info(f"[Epoch {epoch+1}]: Test loss = {avg_test_loss:.4f}")
            
            # 记录详细指标信息
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in test_results.items()])
            logger.info(f"[Epoch {epoch+1}]: Test metrics: {metric_str}")
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
