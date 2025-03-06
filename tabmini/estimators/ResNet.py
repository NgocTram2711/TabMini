import numpy as np
import pandas as pd
import tabmini
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator

# ---------------------- Định nghĩa mô hình TabularResNet ---------------------- #
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity  # residual connection
        out = F.relu(out)
        return out

class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, num_classes, dropout):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.bn_input(out)
        out = F.relu(out)
        out = self.blocks(out)
        logits = self.output_layer(out)
        return logits

class ResNet(BaseEstimator):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_blocks=3,
                 dropout=0.1, epochs=50, batch_size=16, lr=0.001, device="cuda"):
        # Sử dụng CPU (hoặc CUDA nếu có)
        self.device = device if device else get_device()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = TabularResNet(input_dim, hidden_dim, num_blocks, num_classes, dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Nếu dùng CPU, ghi đè hàm kiểm tra CUDA của optimizer
        if self.device == "cpu" or self.device == "cpu":
            self.optimizer._cuda_graph_capture_health_check = lambda: None

        self.result_df = None
        self.best_params_ = None
        self.param_grid = {
            "hidden_dim": [32, 64, 128],
            "num_blocks": [1, 3, 5],
            "dropout": [0.0, 0.1, 0.2],
            "lr": [0.0005, 0.001, 0.01],
            "batch_size": [8, 16, 32],
            "epochs": [10, 25, 50]
            # Các giá trị khác có thể được bật nếu cần
        }

    def fit(self, X, y, X_test, y_test):
        # Chuyển đổi dữ liệu thành tensor
        X_train_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y.to_numpy(), dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        # Tạo dataset và dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        results = []
        best_f1 = -1
        best_model = None

        # Tạo tất cả các tổ hợp tham số
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(
                -1, len(self.param_grid.keys())
            )
        ]

        for param in param_combinations:
            # Khởi tạo mô hình với tham số hiện tại
            hidden_dim = int(param["hidden_dim"])
            num_blocks = int(param["num_blocks"])
            current_model = TabularResNet(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=param["dropout"]
            ).to(self.device)

            print(f"Creating model with: input_dim={self.input_dim}, num_classes={self.num_classes}, hidden_dim={hidden_dim}, num_blocks={num_blocks}")

            # Khởi tạo optimizer và criterion
            optimizer = torch.optim.Adam(current_model.parameters(), lr=param["lr"])
            criterion = torch.nn.CrossEntropyLoss()

            # Huấn luyện mô hình
            current_model.train()
            train_loader = DataLoader(train_dataset, batch_size=int(param["batch_size"]), shuffle=True, drop_last=True)
            for epoch in range(int(param["epochs"])):
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = current_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Đánh giá mô hình trên tập kiểm tra và tính các độ đo bổ sung
            current_model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = current_model(batch_X)
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            # Tính toán các chỉ số: accuracy, f1, precision, recall, auc_roc
            f1 = f1_score(all_labels, all_preds, average="weighted")
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            # Tính AUC dựa trên số lượng class
            if len(np.unique(all_labels)) == 2:
                auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
            else:
                auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr")

            # Lưu kết quả
            results.append({**param, "accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec, "auc_roc": auc})
            print(f"Metrics: accuracy={acc}, f1_score={f1}, precision={prec}, recall={rec}, auc_roc={auc}")

            # Cập nhật mô hình tốt nhất
            if f1 > best_f1:
                best_f1 = f1
                best_model = current_model
                self.best_params_ = param

        # Lưu kết quả vào DataFrame
        self.result_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
        self.model = best_model

        return self

    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)

    def evaluate_model(model, X, y):
        """
        Chia dữ liệu thành 70% train và 30% test, huấn luyện mô hình và tính các chỉ số:
        Accuracy, Precision, Recall, F1-score và AUC.
        """
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        return accuracy, precision, recall, f1, auc
