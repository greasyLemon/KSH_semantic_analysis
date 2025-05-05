import torch
import torch.optim as optim
import torch.nn as nn
import CMT_CLASSIFY
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from CMT_CLASSIFY.dataset.utils import get_dataloader
from CMT_CLASSIFY.model.model import Model
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from CMT_CLASSIFY.dataset.comment_dataset import CommentDataset
from sklearn.model_selection import train_test_split
from CMT_CLASSIFY.dataset.utils import get_cmtwlabel, encode_label
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib

importlib.reload(CMT_CLASSIFY)

train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size=16)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert = AutoModel.from_pretrained("vinai/phobert-base").to(DEVICE)

model = Model(
    phobert=phobert,
    hidden_dim=200,
    kernel_sizes=[4,5,6],
    num_classes=4
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=0.0002241340728327586, weight_decay=9.35376506185603e-05)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter("runs/cmt_classification")

def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs=15):
    model.train()
    
    for epoch in range(num_epochs):
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")  
        
        for batch in progress_bar:
            comment_input = batch["input_ids"].to(device)
            comment_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(comment_input, comment_mask)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{(correct_train / total_train):.4f}"})
        
        train_accuracy = correct_train / total_train
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch+1)

        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device, validation=True)
        writer.add_scalar("Loss/Validation", val_loss, epoch+1)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch+1)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        torch.save(model.state_dict(), f"cmt_classify_epoch_{epoch+1}.pth")

        model.train()

    return val_loss, val_accuracy

def evaluate(model, dataloader, criterion, device, validation=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            comment_input = batch["input_ids"].to(device)
            comment_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(comment_input, comment_mask)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{(correct / total):.4f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    if validation:
        return avg_loss, accuracy
    else:
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

train(model, train_dataloader, val_dataloader, optimizer, criterion, DEVICE, num_epochs=15)
evaluate(model, test_dataloader, criterion, DEVICE)

# def objective(trial):
#     criterion = nn.CrossEntropyLoss()

#     lr = trial.suggest_loguniform("lr", 1e-5, 5e-4)
#     weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 5e-3)
#     hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
#     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
#     kernel_sizes = eval(trial.suggest_categorical("kernel_sizes", ["[2,3,4]", "[3,4,5]", "[3,5,7]"]))

#     full_data, full_labels = get_cmtwlabel()
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     val_accs = []

#     for train_idx, val_idx in skf.split(full_data, full_labels):
#         train_dataloader, val_dataloader, _ = get_dataloader(
#             batch_size=batch_size, use_cv=True, train_idx=train_idx, val_idx=val_idx
#         )

#         model = Model(phobert=phobert, hidden_dim=hidden_dim, kernel_sizes=kernel_sizes, num_classes=4).to(DEVICE)
#         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#         val_loss, val_acc = train(model, train_dataloader, val_dataloader, optimizer, criterion, DEVICE, num_epochs=5)
#         val_accs.append(val_acc)

#     return np.mean(val_accs)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)

# print("Best hyperparameters:", study.best_params)



# import torch
# import torch.nn as nn
# from torchviz import make_dot
# from CMT_CLASSIFY.model.model import Model
# from transformers import AutoModel

# phobert = AutoModel.from_pretrained("vinai/phobert-base")
# hidden_dim = 128
# kernel_sizes = [3, 5, 7]
# num_classes = 4

# model = Model(phobert, hidden_dim, kernel_sizes, num_classes)

# # Tạo dữ liệu đầu vào giả lập
# batch_size = 64
# seq_length = 256
# context_dim = 768

# comment_input = torch.randint(0, 1000, (batch_size, seq_length))
# comment_mask = torch.ones(batch_size, seq_length)
# context_embedding = torch.randn(batch_size, 1, context_dim)

# # Forward pass để tạo output
# output = model(comment_input, comment_mask, context_embedding)
# predicted_label = torch.argmax(output, dim=1)  # Lấy chỉ số lớp có xác suất cao nhất
# print(predicted_label)
# # Vẽ sơ đồ mô hình
# dot = make_dot(output, params=dict(model.named_parameters()))
# dot.render("model_architecture", format="png")

# # Hiển thị sơ đồ trong notebook (nếu dùng Jupyter)
# dot

# from torchview import draw_graph

# # Vẽ sơ đồ mô hình (chỉ hiển thị kiến trúc các layer)
# graph = draw_graph(model, input_data=(comment_input, comment_mask, context_embedding), expand_nested=True)
# graph.visual_graph.render("model_structure", format="pdf")  # Xuất file PDF
