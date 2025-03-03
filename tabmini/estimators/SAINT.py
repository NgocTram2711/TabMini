import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import check_X_y
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# A minimal transformer-inspired model for tabular data
class Model(nn.Module):
    def __init__(self, input_dim, dim=64, depth=3, heads=4, dropout=0.1, num_classes=2):
        super(Model, self).__init__()
        
        # Store parameters
        self.input_dim = int(input_dim)
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        
        # Simple MLP for feature embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim, self.dim)
        )
        
        # Simple MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim * 2, self.num_classes)
        )
        
    def forward(self, x):
        # Process features through MLP
        x = self.mlp(x)
        
        # Classification head
        x = self.classifier(x)
        
        return x

class SAINT(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        time_limit: int = 3600,
        device=None,
        seed: int = 42,
        kwargs: dict = {},
        small_dataset: bool = False,
    ):
        self.time_limit = time_limit
        self.device = device if device else get_device()
        self.seed = seed
        self.kwargs = kwargs
        self.result_df = None
        self.model = None
        self.best_params_ = None
        
        self.param_grid = {
            "depth": [3, 5] if small_dataset else [3, 5, 7],
            "dim": [32, 64] if small_dataset else [64, 128],
            "heads": [2, 4],
            "dropout": [0.1, 0.2, 0.3],
        }

    def fit(self, X, y, X_test, y_test):
        # Ensure data is in the right format
        X_train, y_train = check_X_y(X, y, accept_sparse=False)
        X_test, y_test = np.array(X_test), np.array(y_test)  # Ensure they are NumPy arrays

        # Get number of classes
        num_classes = len(np.unique(y))
        
        # Convert to PyTorch tensors
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
        
        # Explicitly print the shapes to help debug
        print(f"X_train shape: {X_train.shape}")
        print(f"Number of classes: {num_classes}")
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        results = []
        best_f1 = -1
        best_model = None
        
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(-1, len(self.param_grid.keys()))
        ]

        for param in param_combinations:
            # Explicitly cast parameters to integers
            dim = int(param["dim"])
            depth = int(param["depth"])
            heads = int(param["heads"])
            dropout = float(param["dropout"])
            
            print(f"Creating model with: dim={dim}, depth={depth}, heads={heads}, dropout={dropout}")
            
            model = Model(
                input_dim=X_train.shape[1],  # Number of features
                dim=dim,
                depth=depth,
                heads=heads,
                dropout=dropout,
                num_classes=num_classes
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(20):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            y_preds = []
            y_true = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(self.device)
                    
                    out = model(xb)
                    y_pred = torch.argmax(out, dim=1).cpu().numpy()
                    y_preds.extend(y_pred)
                    y_true.extend(yb.numpy())
                    
            # Calculate metrics
            if len(np.unique(y_test)) <= 2:
                f1 = f1_score(y_test, y_pred, average="binary")
            else:
                f1 = f1_score(y_test, y_pred, average="weighted")
                
            acc = accuracy_score(y_true, y_preds)
            results.append({**param, "accuracy": acc, "f1_score": f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                self.best_params_ = param
        
        self.result_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
        self.model = best_model

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