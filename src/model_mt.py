import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class XLMRMultiTask(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", hidden_dropout=0.1):
        super().__init__()
        self.enc = XLMRobertaModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.head_switch = nn.Linear(h, 2)
        self.head_dur = nn.Linear(h, 3)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out.last_hidden_state)  # [B, T, H]
        logits_switch = self.head_switch(x)      # [B, T, 2]
        logits_dur = self.head_dur(x)            # [B, T, 3]
        return logits_switch, logits_dur
