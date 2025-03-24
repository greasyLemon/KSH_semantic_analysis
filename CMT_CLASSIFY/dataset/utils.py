import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from CMT_CLASSIFY.dataset.comment_dataset import CommentDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# data_path = "DATA/labeled_comments_segmented.csv"
data_path = "/home/lamonade/Engineering/KSH_semantic_analysis/DATA/labeled_comments_segmented.csv"

def get_cmtwlabel():
    df = pd.read_csv("/home/lamonade/Engineering/KSH_semantic_analysis/DATA/labeled_comments_segmented.csv")
    comments = df['Comment Text'].tolist()
    labels = df['Label'].tolist()
    return comments, labels

def encode_label(labels: list):
    encoder = LabelEncoder()
    return np.array(encoder.fit_transform(labels))

def get_dataloader(tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")):
    comments, labels = get_cmtwlabel()
    labels = encode_label(labels)

    train_comments, test_comments, train_labels, test_labels = train_test_split(comments, labels, test_size=0.3, train_size=0.7, random_state=42, stratify=labels)

    training_data = CommentDataset(labels=train_labels, texts=train_comments, tokenizer=tokenizer)
    test_data = CommentDataset(labels=test_labels, texts=test_comments, tokenizer=tokenizer)

    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)

    return train_dataloader, test_dataloader