import torch
import torch.optim as optim
import torch.nn as nn
import CMT_CLASSIFY
from CMT_CLASSIFY.dataset.utils import get_dataloader
from CMT_CLASSIFY.model.model import Model
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from CMT_CLASSIFY.dataset.comment_dataset import CommentDataset
from sklearn.model_selection import train_test_split
from CMT_CLASSIFY.dataset.utils import get_cmtwlabel, encode_label
from tqdm import tqdm
import importlib

importlib.reload(CMT_CLASSIFY)

train_dataloader, test_dataloader = get_dataloader()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert = AutoModel.from_pretrained("vinai/phobert-base").to(DEVICE)

# Giả sử bạn có 1 câu context cố định
context_text = "Vào tháng 3/2025, gia đình Kim Sae Ron tiết lộ cô và Kim Soo Hyun đã hẹn hò bí mật từ năm 2015, khi cô mới 15 tuổi, kéo dài đến 2021. Năm 2022, sau vụ tai nạn lái xe, công ty của Kim Soo Hyun đã chi trả 700 triệu won tiền bồi thường cho Kim Sae Ron. Tuy nhiên, năm 2024, cô bất ngờ nhận yêu cầu hoàn trả số tiền này. Cô cố gắng liên lạc với Kim Soo Hyun nhưng không thành công và phát hiện số điện thoại của mình bị lộ cho phóng viên. Gặp khó khăn tài chính và áp lực, Kim Sae Ron đã tự tử vào ngày sinh nhật của Kim Soo Hyun. Gia đình cô tuyên bố sở hữu bằng chứng về mối quan hệ và sẽ công bố dần. Phía Kim Soo Hyun phủ nhận các cáo buộc và tuyên bố sẽ hành động pháp lý đối với thông tin sai lệch."
context_tokens = tokenizer(context_text, return_tensors="pt", padding=True, truncation=True)

# Lấy context embedding (fix cứng)
with torch.no_grad():
    context_embedding = phobert(context_tokens["input_ids"].to(DEVICE), context_tokens["attention_mask"].to(DEVICE))[0]

# Bỏ grad (nếu không muốn fine-tune PhoBERT)
context_embedding = context_embedding.mean(dim=1)  # Lấy mean pool để giảm chiều
context_embedding = context_embedding.to(DEVICE)

model = Model(
    phobert=phobert,
    hidden_dim=128,
    kernel_sizes=[3,4,5],
    num_classes=4
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

def train(model, train_dataloader, optimizer, criterion, context_embedding, device, num_epochs=5):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")  # Tạo progress bar
        
        for batch in progress_bar:
            comment_input = batch["input_ids"].to(device)
            comment_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(comment_input, comment_mask, context_embedding.unsqueeze(0))
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Cập nhật progress bar với loss và accuracy hiện tại
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{(correct / total):.4f}"})
        
        # In kết quả cuối mỗi epoch
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

def evaluate(model, test_dataloader, criterion, context_embedding, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            comment_input = batch["input_ids"].to(device)
            comment_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(comment_input, comment_mask, context_embedding.unsqueeze(0))

            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

train(model, train_dataloader, optimizer, criterion, context_embedding, DEVICE, num_epochs=5)
evaluate(model, test_dataloader, criterion, context_embedding, DEVICE)


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
