import torch
import torch.nn as nn
import torch.nn.functional as F
from CMT_CLASSIFY.model.attention import AttentionLayer

class Model(nn.Module):
    def __init__(self, phobert, hidden_dim, kernel_sizes, num_classes):
        super(Model, self).__init__()

        self.phobert = phobert

        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim * 2,
                kernel_size=k,
                padding=k // 2  # padding = same
            ) for k in kernel_sizes
        ])

        self.context_fc = nn.Linear(768, hidden_dim * 2)

        self.attention = AttentionLayer(hidden_dim=hidden_dim * 2)

        self.fc = nn.Linear(hidden_dim * 2 * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, comment_input, comment_mask, context_embedding):
        with torch.no_grad():
            comment_embedding = self.phobert(comment_input, comment_mask)[0]
        # print(comment_embedding.shape)
        lstm_out, _ = self.lstm(comment_embedding)
        lstm_out = F.relu(lstm_out)
        # print(lstm_out.shape)

        conv_input = torch.permute(lstm_out, (0, 2, 1))
        conv_outputs = [F.relu(conv(conv_input)) for conv in self.convs]

        # for i, vec in enumerate(conv_outputs):
        #     print(f"Shape of conv_out[{i}]: {vec.shape}")

        comment_vector = [torch.max_pool1d(conv, kernel_size=2, stride=2) for conv in conv_outputs]        

        # for i, vec in enumerate(comment_vector):
        #     print(f"Shape of comment_vector[{i}]: {vec.shape}")

        context_vector = self.context_fc(context_embedding)
        # print(context_vector.shape)

        comment_vector = [self.attention(comment, context_vector) for comment in comment_vector]

        comment_vector = torch.cat(comment_vector, dim=1)
        # print("after concat", comment_vector.shape)

        comment_vector = torch.mean(comment_vector, dim=-1)
        logits = self.dropout(self.fc(comment_vector))

        return logits
