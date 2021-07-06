import os
import sys
import logging
import numpy as np
import json
import re
import random

work_dir = os.getcwd()  # 当前路径
sys.path.extend([os.path.abspath(".."), work_dir])

from basic.basic_task import Basic_task, Task_Mode
from basic.register import register_task, find_task
from utils.build_vocab import Vocab
from utils.utils import check_dir

import torch
from torch import nn

from transformers import BertPreTrainedModel, BertConfig, BertTokenizer, BertModel
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

"""
成语完型填空式机器阅读理解任务：
    模型：bert + linear
    数据集：
        Due to data copyright issues，please click the official link to download Chid dataset
        https://drive.google.com/drive/folders/1qdcMgCuK9d93vLVYJRvaSLunHUsGf50u 
    数据比较大，这里只拿5000条数据训练来跑通模型。
"""

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

workdir = os.getcwd()  # 当前路径
project_dir = os.path.split(workdir)[0]

class Config:

    seed = 42   # 随机种子
    gpuids = "0,1"  # 设置显卡序号，若为None，则不使用gpu
    nlog = 100  # 多少step打印一次记录（loss，评估指标）
    early_stop = False

    train_batch_size = 32
    eval_batch_size = 32
    epochs = 5
    lr = 5e-5   # 学习率

    do_train = True
    do_eval = True
    do_infer = False

    # 新增超参数
    margin = 1
    max_len = 128

    task_name = "IdiomCloze"

    # 配置路径
    train_data_path = "/workspace/data/idiom_cloze/train_data.txt"  # 训练集数据的路径，建议绝对路径
    dev_data_path = ["/workspace/data/idiom_cloze/dev_data.txt"]  # 验证集数据的路径，建议绝对路径
    test_data_path = ["/workspace/data/idiom_cloze/test_data.txt"]  # 测试集数据的路径，建议绝对路径

    # transformer结构(Bert, Albert, Roberta等)的预训练模型的配置, 路径也建议是绝对路径
    bert_model_path = "/workspace/Idiom_cloze/pretrained_models/chinese_wwm_pytorch/pytorch_model.bin"  # 预训练模型路径， 例如bert预训练模型
    model_config_path = "/workspace/Idiom_cloze/pretrained_models/chinese_wwm_pytorch/config.json"  # 预训练模型的config文件路径， 一般是json文件
    vocab_path = "/workspace/Idiom_cloze/pretrained_models/chinese_wwm_pytorch/vocab.txt"  # vocab文件路径，可以是预训练模型的vocab.txt文件

    model_save_path = project_dir + f"/model_save/{task_name.lower()}_model"  # 训练过程中最优模型或者训练结束后的模型保存路径
    output_path = project_dir + f"/output/{task_name.lower()}_model"  # 模型预测输出预测结果文件的路径

    # 新增文件路径
    idiom_list_path = "/workspace/data/idiom_cloze/idiomList_process.txt"


# 构建模型动态计算图
class Model(BertPreTrainedModel):
    """
    模型说明：成语完形填空式阅读理解baseline模型
    """
    def __init__(self, model_config, idiom_num):
        super(Model, self).__init__(model_config)
        # 768 is the dimensionality of bert-base-uncased's hidden representations
        # Load the pretrained BERT model
        self.bert = BertModel(config=model_config)
        self.idiom_embedding = nn.Embedding(idiom_num, model_config.hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(model_config.hidden_size, 1)

        self.init_weights()

    def forward(self, inputs):

        input_ids = inputs.get("input_ids", None)
        attention_mask = inputs.get("input_masks", None)
        token_type_ids = inputs.get("token_type_ids", None)
        idiom_ids = inputs.get("idiom_ids", None)
        positions = inputs.get("position", None)
        label = inputs.get("label", None)

        # input_ids [batch, max_seq_length]  encoded_layer [batch, max_seq_length, hidden_state]
        sequence_outputs, pooled_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        blank_states = sequence_outputs[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        encoded_idiom = self.idiom_embedding(idiom_ids)  # [batch, 10， hidden_state]

        multiply_result = torch.einsum('abc,ac->abc', encoded_idiom, blank_states)  # [batch, 10， hidden_state]
        pooled_output = self.dropout(multiply_result)
        logits = self.classifier(pooled_output)
        # logits = self.classifier(multiply_result)  # [batch, 10, 1]
        logits = logits.view(-1, idiom_ids.shape[-1])  # [batch, 10]

        outputs = {
            "logits": logits,
        }
        if label is not  None:
            # # Calculate batch loss based on CrossEntropy
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, label)
            outputs["loss"] = loss
        
        return outputs

# 编写任务
@ register_task
class IdiomCloze(Basic_task):
    def __init__(self, task_config):
        super().__init__(task_config)
        self.task_config = task_config
        self.max_len = task_config.max_len
        # model init 模型初始化，加载预训练模型
        self.model_config = BertConfig.from_pretrained(self.task_config.model_config_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.task_config.vocab_path, lowercase=True)
        self.idiom_vocab = Vocab(self.task_config.idiom_list_path)

        self.model = Model.from_pretrained(pretrained_model_name_or_path=self.task_config.bert_model_path,
                                           config=self.model_config, idiom_num=self.idiom_vocab.vocab_size)

        if self.task_config.gpuids != None:
            self.model.to(self.device)
        # 单机多卡训练
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def evaluate(self, dataset, mode=Task_Mode.Eval, epoch=None):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.task_config.eval_batch_size,
            num_workers=0
        )
        self.model.eval()
        pred_labels = []
        true_labels = []
        loss_buffer = 0
        for bi, batch in enumerate(data_loader):
            outputs = self.run_one_step(batch, self.model)
            logits = outputs.pop("logits")
        
            prob_outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(prob_outputs, axis=1)

            pred_labels.extend(pred_label.tolist())
            if mode == Task_Mode.Eval:
                loss = outputs.pop("loss")
                loss = loss.mean()
                loss_buffer += loss.item()
                label = batch["label"].cpu()
                true_labels.extend(label.numpy().tolist())
        if mode == Task_Mode.Eval:
            total_acc = accuracy_score(true_labels, pred_labels)
            logger.info("Evaluate: epoch={}, step={}, acc = {:0.4f}".format(epoch, self.global_step, total_acc))
            return total_acc
        else:
            return pred_labels

    def train(self, dataset, valid_dataset=None):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.task_config.train_batch_size,
            num_workers=0
        )
        num_train_steps = int(len(dataset) / self.task_config.train_batch_size * self.task_config.epochs)
        optimizer, scheduler = self.create_optimizer(self.model, use_scheduler=True, num_warmup_steps=1000,
                                                     num_train_steps=num_train_steps)
        self.model.train()
        # Train the model on each batch
        # Reset gradients
        loss_buffer = 0
        for epoch in range(self.task_config.epochs):
            for bi, batch in enumerate(data_loader):
                self.model.zero_grad()
                outputs = self.run_one_step(batch, self.model)
                logits = outputs.pop("logits")
                loss = outputs.pop("loss")
                # Calculate gradients based on loss
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   #更新模型参数
                scheduler.step()  # 更新learning rate
                self.global_step += 1

                loss_buffer += loss.item()
                if self.global_step % self.task_config.nlog == 0:
                    logger.info("epoch={}, step={}, loss={:.4f}".format(epoch+1, self.global_step, loss_buffer / self.task_config.nlog))
                    loss_buffer = 0
                
            if valid_dataset != None:
                eval_acc = self.evaluate(valid_dataset, mode=Task_Mode.Eval, epoch=epoch+1)
                self.model.train()
                if self.task_config.early_stop:
                    self.es(epoch, eval_acc, self.model, model_path=self.task_config.model_save_path)
                    if self.es.early_stop:
                        logger.info("********** Early stopping ********")
                        break
    
    def read_data(self, file, mode):
        """
        根据不同任务编写数据处理，建议将原始数据进行预处理之后再在这里写数据处理成模型输入结构
        """
        dataset = []
        with open(file, "r", encoding="utf-8") as fin:
            data_id = 100000000
            lines = fin.readlines()[:5000]
            tk0 = tqdm(lines, total=len(lines))
            for line in tk0:
                cur_data = json.loads(line)
                groundTruth = cur_data["groundTruth"]
                candidates = cur_data["candidates"]
                content = cur_data["content"]
                realCount = cur_data["realCount"]
                for i in range(realCount):
                    content = content.replace("#idiom#", f"#idiom{i+1}#", 1)
                tags = re.findall("#idiom\d+#", content)
                for tag in tags:
                    tmp_context = content
                    for other_tag in tags:
                        if other_tag != tag:
                            tmp_context = tmp_context.replace(other_tag, self.tokenizer.unk_token)

                    feature_id = int(tag[6: -1])
                    left_part, right_part = re.split(tag, tmp_context)
                    left_ids = self.tokenizer.encode(left_part, add_special_tokens=False)
                    right_ids = self.tokenizer.encode(right_part, add_special_tokens=False)

                    half_length = int(self.max_len / 2)
                    if len(left_ids) < half_length:  # cut at tail
                        st = 0
                        ed = min(len(left_ids) + 1 + len(right_ids), self.max_len - 2)
                    elif len(right_ids) < half_length:  # cut at head
                        ed = len(left_ids) + 1 + len(right_ids)
                        st = max(0, ed - (self.max_len - 2))
                    else:  # cut at both sides
                        st = len(left_ids) + 3 - half_length
                        ed = len(left_ids) + 1 + half_length

                    text_ids = left_ids + [self.tokenizer.mask_token_id] + right_ids
                    input_ids = [self.tokenizer.cls_token_id] + text_ids[st:ed] + [self.tokenizer.sep_token_id]

                    position = input_ids.index(self.tokenizer.mask_token_id)

                    token_type_ids = [0] * len(input_ids) + [0] * (self.max_len - len(input_ids))
                    input_masks = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
                    input_ids = input_ids + [0] * (self.max_len - len(input_ids))

                    label = candidates[i].index(groundTruth[i])
                    idiom_ids = [self.idiom_vocab.word2id[each] for each in candidates[i]]

                    assert len(input_ids) == self.max_len
                    assert len(input_masks) == self.max_len
                    assert len(token_type_ids) == self.max_len

                    # Return the processed data where the lists are converted to `torch.tensor`s
                    dataset.append({
                        'data_id': torch.tensor(data_id, dtype=torch.long),
                        'feature_id': torch.tensor(feature_id, dtype=torch.long),
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'input_masks': torch.tensor(input_masks, dtype=torch.long),
                        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                        'idiom_ids': torch.tensor(idiom_ids, dtype=torch.long),
                        'label': torch.tensor(label, dtype=torch.long),
                        'position': torch.tensor(position, dtype=torch.long)
                    })
                    data_id += 1
        
        return dataset


def seed_set(seed):
    '''
    set random seed of cpu and gpu
    :param seed:
    :param n_gpu:
    :return:
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run():
    config = Config()
    check_dir([config.model_save_path, config.output_path])
    seed_set(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuids  # 设置gpu序号
    task_cls = find_task(config.task_name)
    task = task_cls(task_config=config)
    if config.do_train:
        dataset = task.read_data(config.train_data_path, mode=Task_Mode.Train)
        if config.do_eval:
            valid_dataset = task.read_data(config.dev_data_path[0], mode=Task_Mode.Eval)
            task.train(dataset, valid_dataset=valid_dataset)
        else:
            task.train(dataset)
    if config.do_eval:
        task.load_model(config.model_save_path)
        for dev_path in config.dev_data_path:
            logging.info(f"Evaluating model in {dev_path}")
            dataset = task.read_data(dev_path, mode=Task_Mode.Eval)
            logging.info(f"dev dataset size = {len(dataset)}")
            task.evaluate(dataset, mode=Task_Mode.Eval)
    if config.do_infer:
        task.load_model(config.model_save_path)
        for test_path in config.test_data_path:
            logging.info(f"Testing model in {test_path}")
            dataset = task.read_data(test_path, mode=Task_Mode.Infer)
            logging.info(f"test dataset size = {len(dataset)}")
            task.evaluate(dataset, mode=Task_Mode.Infer)

if __name__ == '__main__':
    run()