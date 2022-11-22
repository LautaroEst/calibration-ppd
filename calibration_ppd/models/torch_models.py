import torch
from torch import nn


class CBOW(nn.Module):

    def __init__(self,tokenizer,hidden_size,output_size):
        super().__init__()
        num_embeddings = len(tokenizer)
        self.linear_input = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=hidden_size,
            padding_idx=tokenizer.pad_token_id
        )
        self.linear_output = nn.Linear(in_features=hidden_size,out_features=output_size)

    def forward(self,input_ids,token_type_ids,attention_mask):
        x = self.linear_input(input_ids).mean(dim=1)
        x = torch.relu(x)
        logits = self.linear_output(x)
        return {"logits": logits}
        
