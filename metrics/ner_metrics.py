from collections import Counter
import re
import numpy as np

class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()
        self.outputs = []

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def eval_result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_tags, pred_tags):
        '''
        :param label_tags:
        :param pred_tags:
        :return:
        Example:
            >>> label_tags = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_tags = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_tag, pre_tag in zip(label_tags, pred_tags):
            label_entities = get_entities(label_tag, self.id2label,self.markup)
            pre_entities = get_entities(pre_tag, self.id2label,self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
            
    def get_entity(self, pred_tags):
        '''
        Example:
            >>> pred_tags = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        pre_entities = get_entities(pred_tags, self.id2label, self.markup)
        return pre_entities
        

class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])

def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq, id2label=None):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)




def get_ner_metrics(true_labels, pred_labels):
    # true_labels 和 pred_labels 为 string 组成的 list
    # 每个 string 的形式为 'O O O O B-电话号码 I-电话号码 I-电话号码 E-电话号码'
    # 返回 slot_macro_f1(type1), slot_macro_f1(type2), slot_micro_f1, illegal_slot_rate


    true_tags = [label.split(' ') for label in true_labels]
    pred_tags = [label.split(' ') for label in pred_labels]


    def get_slots_v2(true_tag, pred_tag):
        # v1 在计算 pred_slot 类型的时候，有问题
        # v2 当 predict slot 里面的所有 tag 和 true slot 里面的所有 tag 一致，才算是 slot 类别预测正确
        #    当 predict slot 有些 tag 和 true slot 里面的所对应的 tag 不一致，则选择出现次数最多的 tag 所对应的类别，
        #    作为 predict slot 的 slot 类别，这个类别不可以是 true slot 的类别，也不可以是 '0'

        # 去除原来 la 的 '0' 标签
        zero_index = [i for i, tag in enumerate(true_tag) if tag == 'O']
        _ = [true_tag.pop(index - i) for i, index in enumerate(zero_index)]
        _ = [pred_tag.pop(index - i) for i, index in enumerate(zero_index)]

        if len(true_tag) == 0:
            # true_tag 都是 'O' ，去除该样本
            return [], []

        true_slot = []
        pred_slot = []
        cur_slot = []
        cur_indexes = []
        for index, tag in enumerate(true_tag):
            if tag[0] == 'B' and (index == len(true_tag) - 1):
                # 该 slot 只有一个字，且位于句尾
                slot_type = tag.split('-')[-1]
                p_slot = pred_tag[index]
                if p_slot == 'O':
                    continue
                else:
                    p_slot_type = p_slot.split('-')[-1]
                true_slot.append(slot_type)
                pred_slot.append(p_slot_type)

            elif tag[0] == 'B' and tag.split('-')[-1] != true_tag[index + 1].split('-')[-1]:
                # 该 slot 只有一个字，且非位于句尾
                slot_type = tag.split('-')[-1]
                p_slot = pred_tag[index]
                if p_slot == 'O':
                    continue
                else:
                    p_slot_type = p_slot.split('-')[-1]
                true_slot.append(slot_type)
                pred_slot.append(p_slot_type)

            elif 'E' in tag:
                # 提取完一个 slot
                cur_slot.append(tag)
                cur_indexes.append(index)

                slot_type = list(set([s.split('-')[-1] for s in cur_slot]))
                if len(slot_type) != 1:
                    # 如果 true slot 里面存在多个 slot type 则报错，数据标注问题
                    raise Exception('len(slot_type) != 1')
                else:
                    slot_type = slot_type[0]

                p_cur_slot = [pred_tag[i] for i in cur_indexes]
                if p_cur_slot == cur_slot:
                    # predict slot 里面的所有 tag 和 true slot 里面的所有 tag 一致，才算是 slot 类别预测正确
                    p_slot_type = slot_type
                else:
                    # predict slot 有些 tag 和 true slot 里面的所对应的 tag 不一致
                    # 选择出现最多，且不是 slot_type 和 '0' 的 slot 类型
                    p_slot_type = []
                    for s in p_cur_slot:
                        if s == 'O':
                            p_slot_type.append(s)
                        else:
                            p_slot_type.append(s.split('-')[-1])
                    count_dict = {}
                    for s in p_slot_type:
                        count_dict[s] = count_dict.get(s, 0) + 1
                    count_list = [(k, v) for k, v in count_dict.items() if k != slot_type and k != 'O']
                    if len(count_list) != 0:
                        count_list.sort(key=lambda x: x[1], reverse=True)
                        p_slot_type = count_list[0][0]
                    else:
                        # 除了 slot_type 就是 'O'，所以 抛弃这个 slot
                        cur_slot = []
                        cur_indexes = []
                        continue
                true_slot.append(slot_type)
                pred_slot.append(p_slot_type)

                cur_slot = []
                cur_indexes = []

            else:
                cur_slot.append(tag)
                cur_indexes.append(index)

        if len(true_slot) != len(pred_slot):
            raise Exception('len(true_slot) != len(pred_slot)')

        return true_slot, pred_slot
    
    def cal_macro_f1(true_labels, pred_labels, cs):
        def get_TP_FP_FN(true_labels, pred_labels, c):
            TP = 0
            FP = 0
            FN = 0
            for t_label, p_label in zip(true_labels, pred_labels):
                if t_label == c:
                    if t_label == p_label:
                        TP += 1
                    if t_label != p_label:
                        FN += 1
                elif p_label == c:
                    if t_label != p_label:
                        FP += 1
                else:
                    pass
            return TP, FP, FN

        Ps = []
        Rs = []
        Fs = []
        for c in cs:
            P = 0
            R = 0
            F1 = 0
            TP, FP, FN = get_TP_FP_FN(true_labels, pred_labels, c)
            if TP + FP != 0:
                P = TP / (TP + FP)

            if TP + FN != 0:
                R = TP / (TP + FN)

            if P + R != 0:
                F1 = 2 * (P * R) / (P + R)

            Ps.append(P)
            Rs.append(R)
            Fs.append(F1)

        macro1_f1 = np.mean(Fs)
        macro_p = np.mean(Ps)
        macro_r = np.mean(Rs)
        if macro_p + macro_r != 0:
            macro2_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
        else:
            macro2_f1 = 0

        return macro1_f1, macro2_f1
        
    def cal_micro_f1(true_labels, pred_labels, cs):
        def get_TP_FP_FN(true_labels, pred_labels, c):
            TP = 0
            FP = 0
            FN = 0
            for t_label, p_label in zip(true_labels, pred_labels):
                if t_label == c:
                    if t_label == p_label:
                        TP += 1
                    if t_label != p_label:
                        FN += 1
                elif p_label == c:
                    if t_label != p_label:
                        FP += 1
                else:
                    pass
            return TP, FP, FN

        all_TP = 0
        all_FP = 0
        all_FN = 0
        for c in cs:
            TP, FP, FN = get_TP_FP_FN(true_labels, pred_labels, c)
            all_TP += TP
            all_FP += FP
            all_FN += FN

        if all_TP + all_FP == 0:
            micro_p = 0
        else:
            micro_p = all_TP / (all_TP + all_FP)
        if all_TP + all_FN == 0:
            micro_r = 0
        else:
            micro_r = all_TP / (all_TP + all_FN)
        if micro_p + micro_r != 0:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        else:
            micro_f1 = 0
        return micro_f1

    def check_have_illegal_slot(pred_label):
        # pred_label = 'O O B-歌曲 I-歌曲 E-歌曲 O O O B-歌曲 I-歌曲 I-歌曲 I-歌曲 E-歌曲'

        if len(re.findall(r'O\s(?:[I|E]\-[^\s]+\s)+O', pred_label)) != 0:
            return True

        if len(re.findall(r'^(?:[I|E]\-[^\s]+\s)+O', pred_label)) != 0:
            return True

        if len(re.findall(r'O\s(?:[I|E]\-[^\s]+\s)*(?:[I|E]\-[^\s]+)$', pred_label)) != 0:
            return True

        if len(re.findall(r'^(?:[I|E]\-[^\s]+\s)*(?:[I|E]\-[^\s]+)$', pred_label)) != 0:
            return True

        items0 = re.findall(r'O\sB\-([^\s]+)\s(I\-[^\s]+\s(?:I\-[^\s]+\s)*)*E\-([^\s]+)\sO', pred_label)
        items1 = re.findall(r'^B\-([^\s]+)\s(I\-[^\s]+\s(?:I\-[^\s]+\s)*)*E\-([^\s]+)\sO', pred_label)
        items2 = re.findall(r'O\sB\-([^\s]+)\s(I\-[^\s]+\s(?:I\-[^\s]+\s)*)*E\-([^\s]+)$', pred_label)
        items3 = re.findall(r'^B\-([^\s]+)\s(I\-[^\s]+\s(?:I\-[^\s]+\s)*)*E\-([^\s]+)$', pred_label)
        items = items0 + items1 + items2 + items3
        for item in items:
            if len(item) == 3:
                if item[1] == '':
                    if len(set([item[0], item[2]])) != 1:
                        return True
                else:
                    info = re.findall('I\-([^\s]+)', item[1])
                    if len(info) != 0:
                        if len(set([item[0], item[2]] + info)) != 1:
                            return True
            else:
                raise Exception('error')
            

        return False
        
    true_slots = []
    pred_slots = []
    for pred_tag, true_tag in zip(pred_tags, true_tags):
        true_slot, pred_slot = get_slots_v2(true_tag, pred_tag)
        if true_slot == []:
            pass
        else:
            true_slots.extend(true_slot)
            pred_slots.extend(pred_slot)

    def build_ner_feature_map(ner_train_pth):
        labels = []
        with open(ner_train_pth, 'r', encoding='utf-8') as fp:
            for index, line in enumerate(fp.readlines()):
                if index == 0:
                    continue
                label = line.split('\t')[-1].strip()
                tags = [tag.strip() for tag in label.split('\x02')]
                labels.extend(tags)
        tags_uni = list(set(labels))
        tags_uni.sort(key=lambda x: x)
        feature_map = {tag: index for index, tag in enumerate(tags_uni)}
        return feature_map

    slot_macro_f1 = cal_macro_f1(true_labels=true_slots, pred_labels=pred_slots, cs=list(set(true_slots)))
    slot_micro_f1 = cal_micro_f1(true_labels=true_slots, pred_labels=pred_slots, cs=list(set(true_slots)))

    illegal_num = 0
    for index, pred_label in enumerate(pred_labels):
        if check_have_illegal_slot(pred_label):
            illegal_num += 1
    illegal_slot_rate = illegal_num / len(pred_labels)

    return slot_macro_f1[0], slot_macro_f1[1], slot_micro_f1, illegal_slot_rate