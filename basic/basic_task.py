import os
from torch.functional import Tensor
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW 
import torch

from utils.EarlyStopping import EarlyStopping


class Task_Mode(object):
    Train = 1  # 训练模式
    Eval = 2   # 评估模式，验证集和测试集有标注结果
    Infer = 3  # 推理模式，测试集无标注结果


class Basic_task(object):
    def __init__(self, task_config):
        super(Basic_task, self).__init__()
        self.task_config = task_config
        self.buffer_loss = 0
        self.global_step = 0

        if self.task_config.gpuids != None:
            # Set device as `cuda` (GPU)
            # 使用gpu训练
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()

        if task_config.early_stop:
            # 提前停止训练，减少过拟合风险
            self.es = EarlyStopping(patience=2, mode="max", delta=0.00001)

    def create_optimizer(self, model, use_scheduler=True, num_warmup_steps=1000, num_train_steps=0):
        """
        创建优化器optimizer和scheduler，默认使用AdamW优化器, 和线性warmup
        """
        param_optimizer = list(model.named_parameters())
        # Specify parameters where weight decay shouldn't be applied
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # Define two sets of parameters: those with weight decay, and those without
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        # Instantiate AdamW optimizer with our two sets of parameters, and a learning rate of 3e-5
        optimizer = AdamW(optimizer_parameters, lr=self.task_config.lr)
        if use_scheduler:
            # Create a scheduler to set the learning rate at each training step
            # "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period." (https://pytorch.org/docs/stable/optim.html)
            # Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
            return optimizer, scheduler
        else:
            return optimizer

    def run_results(self, model, data_loader, pred_outputs):
        pass

    def evaluate(self, dataset, mode=Task_Mode.Eval, epoch=None):
        """
        评估函数
        :param dataset:
        :param mode:
        :param epoch:
        :return:
        """
        pass

    def train(self, dataset, valid_dataset=None):
        """
        训练函数，修改部分较少
        :param dataset:
        :param valid_dataset:
        :return:
        """
        pass

    def read_data(self, file, mode):
        """
        数据处理，需要自己编写
        :param file:
        :param mode:
        :return:
        """
        pass

    def run_one_step(self, batch, model):
        inputs = {}
        if self.task_config.gpuids != None:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)   # 数据加载到显存中
        # Use ids, masks, and token types as input to the model
        # Predict logits for each of the input tokens for each batch
        outputs = model(inputs)  #

        return outputs
    
    def predict(self, model, data_loader):
        model.eval()
        outputs = []
        for bi, batch in enumerate(data_loader):
            model_outputs = self.run_one_step(batch, model)
            for k, v in model_outputs.items():
                if isinstance(v, torch.Tensor):
                    model_outputs[k] = v.detach().cpu().numpy().tolist()
            keys = list(batch.keys())
            batch_size = len(batch[keys[0]])
            for i in range(batch_size):
                item_output = {k: v[i] for k, v in batch.items()}
                item_output.update({k: v[i] for k, v in model_outputs.items() if not k.endswith("loss")})   
                outputs.append(item_output)  
        return outputs

    def loss_buffer(self, loss):
        self.buffer_loss += loss.item()

    def save_checkpoint(self, model, model_path, epoch=None, save_step=False):
        if epoch is not None:
            model_path = model_path + "_" + epoch
        if save_step:
            model_path = model_path + "_" + self.global_step
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
      
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(model_path)
    
    def load_model(self, model_path):
        trained_model_path = os.path.join(model_path, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(trained_model_path))
        
