import torch
import torch.nn as nn
from transformers import AutoModel

class BiomedBERTProjector(nn.Module):
    def __init__(self, pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT', d_model=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  
        self.projector = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states[-4:]  
        last_hidden = torch.stack(all_hidden_states, dim=0).mean(dim=0) 
        return self.projector(last_hidden)