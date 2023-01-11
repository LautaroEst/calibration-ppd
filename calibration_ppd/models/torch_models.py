import torch
from torch import nn
from math import sqrt

class CBOW(nn.Module):

    def __init__(self,num_embeddings,hidden_size,output_size,pad_idx,random_seed):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_idx = pad_idx
        self.linear_input = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=hidden_size,
            padding_idx=pad_idx
        )
        self.linear_output = nn.Linear(in_features=hidden_size,out_features=output_size)
        self.init_params(random_seed)

    def forward(self,input_ids,token_type_ids,attention_mask):
        x = self.linear_input(input_ids).mean(dim=1)
        x = torch.relu(x)
        logits = self.linear_output(x)
        return {"logits": logits}

    def init_params(self,random_seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        embeddings_weights = torch.randn(tuple(self.linear_input.weight.size()),generator=generator)
        linear_weights = torch.rand(tuple(self.linear_output.weight.size()),generator=generator) * 2 * sqrt(self.hidden_size) - sqrt(self.hidden_size)
        linear_bias = torch.rand(tuple(self.linear_output.bias.size()),generator=generator) * 2 * sqrt(self.output_size) - sqrt(self.output_size)

        with torch.no_grad():
            self.linear_input.weight.copy_(embeddings_weights)
            self.linear_input.weight[self.pad_idx,:].fill_(0.)
            self.linear_output.weight.copy_(linear_weights)
            self.linear_output.bias.copy_(linear_bias)