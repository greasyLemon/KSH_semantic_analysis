import torch 
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, comment_vector, context_vector):
        # print((self.W(context_vector)).shape)
        attn_weights = torch.tanh(self.W(context_vector))
        attn_weights = attn_weights.permute(0, 2, 1)
        # print("attention", attn_weights.shape, comment_vector.shape)
        fused_vector = torch.mul(attn_weights, comment_vector)

        return fused_vector

