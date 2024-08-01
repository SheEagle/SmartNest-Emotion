import torch
import torch.nn as nn


from transformers import BertTokenizer, BertModel

__all__ = ['BertTextEncoder']


class BertTextEncoder(nn.Module):
    def __init__(self, language='cn', use_finetune=False, device='cpu'):
        """
        language: en / cn
        初始化方法，接受language参数，表示语言是英文('en')还是中文('cn')。
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        # 使用BertTokenizer和BertModel类
        if language == 'en':
            # 如果语言是英文，使用英文的Bert预训练模型
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_model/bert_en')
        elif language == 'cn':
            # 如果语言是中文，使用中文的Bert预训练模型
            self.tokenizer = tokenizer_class.from_pretrained('bert-base-chinese', use_auth_token=True,
                                                             cache_dir="../../pre_trained")
            self.model = model_class.from_pretrained('bert-base-chinese', cache_dir="../../pre_trained").to(device)

        self.use_finetune = use_finetune
        # 是否使用微调（fine-tuning）的标志位

    def get_tokenizer(self):
        """
        获取Bert的tokenizer对象。
        """
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        从原始文本数据中获取Bert的输入id（input_ids）。
        """
        input_ids = self.get_id(text)
        input_ids = input_ids.to(self.device)  # 将input_ids移动到与模型相同的设备上
        with torch.no_grad():
            # 禁用梯度计算
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
            # 获取Bert模型的最后一层隐藏状态
        return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        前向传播方法，接受包含文本信息的张量，返回Bert模型的输出。
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            # 如果使用微调，允许梯度计算
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
            # 获取Bert模型的最后一层隐藏状态
        else:
            with torch.no_grad():
                # 禁用梯度计算
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
                # 获取Bert模型的最后一层隐藏状态
        return last_hidden_states


if __name__ == "__main__":
    bert_normal = BertTextEncoder()
