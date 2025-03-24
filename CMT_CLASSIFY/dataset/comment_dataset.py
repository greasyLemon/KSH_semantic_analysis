import torch
from torch.utils.data import Dataset
from CMT_CLASSIFY.config import TOKENIZER_MAX_LEN

class CommentDataset(Dataset):
    def __init__(self, labels, texts, tokenizer):
        self.labels = labels
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = TOKENIZER_MAX_LEN

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        texts = self.texts[index]
        labels = self.labels[index]

        encoding = self.tokenizer(texts, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }