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
    return "cuda" if torch.cuda.is_available() else "cpu"

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
    def __init__(self, input_dim, hidden_dim=128, lr=0.001, epochs=10, batch_size=32, dropout=0.2, output_dim=1, num_heads=4, num_layers=2):
        super(Model, self).__init__()

        self.device = get_device()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        # Model architecture
         # Embedding layer
        print(f"input_dim{input_dim}")
        print(f"hidden_dim{hidden_dim}")
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Transformer blocks for Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Inter-Sample Attention (Modeled as an extra attention layer)
        self.inter_sample_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )


    def forward(self, X):
        # Self-Attention
        X = torch.clamp(X, min=0, max=self.embedding.num_embeddings - 1)
        
        X = X.to(self.device)
        X = X.long()
        x = self.embedding(X)
        x = self.relu(x)
        x = self.self_attention(x)

        # Inter-Sample Attention
        x, _ = self.inter_sample_attention(x, x, x)
        x = x.squeeze(1)

        x = self.dropout_layer(x)
        # Final classification
        return self.fc(x)
    
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.long).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(self.device)

        self.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for data, target in train_loader:
                optimizer.zero_grad()
                target = target.view(-1, 1)
                output = self(data)  # Forward pass
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.long)
        self.eval()
        with torch.no_grad():
            output = self(X_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

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
            "epochs": [300],
            "batch_size": [32, 64],
            "dropout": [0.2, 0.3]
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
        
        # Explicitly print the shapes to help debug
        print(f"X_train shape: {X_train.shape}")
        print(f"Number of classes: {num_classes}")
        
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

            print(f"Training model with: {params}")
            model = Model(self.input_dim, hidden_dim=hidden_dim, batch_size=batch_size, lr=lr, epochs=epochs, dropout=dropout).to(self.device)
            
            model.fit(X_train, y_train)

            model.eval()
            print(model.predict(X_test))

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_prob = model.predict(X_test)
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            best_threshold = thresholds[np.argmax(precision * recall)] if len(thresholds) > 0 else 0.5
            y_pred = (y_prob > best_threshold).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=1, average='weighted')

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=1),
                "recall": recall_score(y_test, y_pred, zero_division=1),
                "f1_score": f1,
                "auc": roc_auc_score(y_test, y_prob)
            }
            results.append(metrics)
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
            self.result_df.to_csv(filename, index=False)