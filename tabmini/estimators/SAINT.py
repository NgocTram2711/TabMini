import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import check_X_y
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def get_device():
    return "cpu"
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm(x + ff_output)
        return x


class IntersampleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm(x + ff_output)
        return x

# A minimal transformer-inspired model for tabular data
class Model(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128,  lr=0.001, epochs=10, batch_size=32, dropout=0.2, output_dim=1, num_heads=4, num_layers=2):        
        super(Model, self).__init__()
        self.device = get_device()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        # Model architecture
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer blocks for Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Inter-Sample Attention (Modeled as an extra attention layer)
        self.inter_sample_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Self-Attention
        x = self.embedding(x)
        x = self.self_attention(x.unsqueeze(1)).squeeze(1)

        # Inter-Sample Attention
        x, _ = self.inter_sample_attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1)

        # Final classification
        return self.fc(x)
    
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.long).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        train_loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for data, target in train_loader:
                optimizer.zero_grad()
                target = target.view(-1, 1)
                output = self(data)  # Forward pass
                loss = criterion(output, target.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()


    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print(f'Model saved to {filename}')

class SAINT(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim, 
        time_limit: int = 3600,
        device=None,
        seed: int = 42,
        small_dataset: bool = False,
    ):
        self.time_limit = time_limit
        self.device = device if device else get_device()
        self.seed = seed
        self.result_df = None
        self.model = None
        self.best_params_ = None
        self.input_dim = input_dim
        
        self.param_grid = {
            "hidden_dim": [128, 256],
            "lr": [0.001, 0.0005],
            "epochs": [10, 20],
            "batch_size": [32, 64],
            "dropout": [0.2, 0.3],
            "num_heads": [ 4, 8],
            "num_layers": [2, 4, 6]
        }

    def fit(self, X, y, X_test, y_test):
        # Ensure data is in the right format
        X_train, y_train = check_X_y(X, y, accept_sparse=False)
        X_test, y_test = np.array(X_test), np.array(y_test)  # Ensure they are NumPy arrays

        # Get number of classes
        num_classes = len(np.unique(y))
        
        # Convert to PyTorch tensors
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(self.device), torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        # Create hyperparameter combinations
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(-1, len(self.param_grid.keys()))
        ]

        results = []
        best_f1 = -1
        best_model = None
        
        for params in param_combinations:
            # Extract parameters
            hidden_dim = int(params["hidden_dim"])
            lr = float(params["lr"])
            epochs = int(params["epochs"])
            batch_size = int(params["batch_size"])
            dropout = float(params["dropout"])

            model = Model(self.input_dim, num_classes, hidden_dim=hidden_dim, batch_size=batch_size, lr=lr, epochs=epochs, dropout=dropout).to(self.device)
            
            model.fit(X_train, y_train)

            model.eval()

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_prob = model(X_test_tensor).detach().numpy().flatten()
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            best_threshold = thresholds[np.argmax(precision * recall)] if len(thresholds) > 0 else 0.5
            y_pred = (y_prob > best_threshold).astype(int)

            # if len(thresholds) > 0:
            #     # Tính f1 score cho từng threshold (bỏ phần tử cuối của precision và recall)
            #     f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
            #     best_threshold = thresholds[np.argmax(f1_scores)]
            # else:
            #     best_threshold = 0.5

            # y_pred = (y_prob > best_threshold).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=1)
            accuracy = accuracy_score(y_test, y_pred)  # Loại bỏ dấu phẩy thừa
            auc = roc_auc_score(y_test, y_prob)

            results.append({
                **params,
                "accuracy": accuracy,
                "f1_score": f1,
                "auc_roc": auc
            })
            print(f"f1 {f1}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                self.best_params_ = params
        
        # Store results
        self.model = best_model
        self.result_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)

    def predict(self, X):
        """
        Predict class labels for X.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.cpu().numpy()

    def save_results(self, filename):
        if self.result_df is not None:
            print("saved")
            self.result_df.to_csv(filename, index=False)