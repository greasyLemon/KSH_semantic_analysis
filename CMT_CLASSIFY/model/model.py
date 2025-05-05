import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, phobert, hidden_dim, kernel_sizes, num_classes):
        super(Model, self).__init__()

        self.phobert = phobert

        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True,
            dropout=0.3
        )
        self.bert_norm = nn.LayerNorm(768)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim * 2,
                kernel_size=k,
                padding=k // 2,  # padding = same
                groups=hidden_dim * 2
            ) for k in kernel_sizes
        ])

        self.layernorm = nn.LayerNorm(hidden_dim * 2)

        self.fc = nn.Linear(hidden_dim * 2 * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, comment_input, comment_mask):
        with torch.no_grad():
            comment_embedding = self.phobert(comment_input, comment_mask)[0]
        comment_embedding = self.bert_norm(comment_embedding)

        lstm_out, _ = self.lstm(comment_embedding)
        lstm_out = self.dropout(F.gelu(lstm_out))
        lstm_out = F.gelu(lstm_out)
        lstm_out = self.layernorm(lstm_out)

        conv_input = torch.permute(lstm_out, (0, 2, 1))
        conv_outputs = [F.gelu(conv(conv_input)) for conv in self.convs]
        conv_outputs = [self.dropout(conv) for conv in conv_outputs]

        comment_vector = [F.adaptive_avg_pool1d(conv, 1) for conv in conv_outputs]

        comment_vector = torch.cat(comment_vector, dim=1)
        comment_vector = torch.mean(comment_vector, dim=-1)

        logits = self.dropout(self.fc(comment_vector))
        return logits