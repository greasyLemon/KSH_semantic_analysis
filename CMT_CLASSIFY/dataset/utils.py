import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from CMT_CLASSIFY.dataset.comment_dataset import CommentDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# data_path = "DATA/labeled_comments_segmented.csv"
data_path = "/kaggle/working/DATA/labeled_comments_segmented.csv"

def get_cmtwlabel():
    df = pd.read_csv("/kaggle/working/DATA/labeled_comments_segmented.csv")
    comments = df['Comment Text'].tolist()
    labels = df['Label'].tolist()
    return comments, labels

def encode_label(labels: list):
    encoder = LabelEncoder()
    return np.array(encoder.fit_transform(labels))

from sklearn.model_selection import train_test_split

def get_dataloader(tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base"), batch_size=64, 
                   use_cv=False, train_idx=None, val_idx=None):
    comments, labels = get_cmtwlabel()
    labels = encode_label(labels)

    if use_cv and train_idx is not None and val_idx is not None:
        train_comments, train_labels = [comments[i] for i in train_idx], [labels[i] for i in train_idx]
        val_comments, val_labels = [comments[i] for i in val_idx], [labels[i] for i in val_idx]
        test_comments, test_labels = [], []
    else:
        train_comments, temp_comments, train_labels, temp_labels = train_test_split(
            comments, labels, test_size=0.3, train_size=0.7, random_state=42, stratify=labels
        )

        val_comments, test_comments, val_labels, test_labels = train_test_split(
            temp_comments, temp_labels, test_size=0.5, train_size=0.5, random_state=42, stratify=temp_labels
        )

    train_data = CommentDataset(labels=train_labels, texts=train_comments, tokenizer=tokenizer)
    val_data = CommentDataset(labels=val_labels, texts=val_comments, tokenizer=tokenizer)
    test_data = CommentDataset(labels=test_labels, texts=test_comments, tokenizer=tokenizer) if test_comments else None

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4) if test_data else None

    return train_dataloader, val_dataloader, test_dataloader