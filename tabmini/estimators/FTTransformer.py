import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from rtdl import FTTransformer


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
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
        
        self.param_grid = {
            "depth": [3, 5] if small_dataset else [3, 5, 7],
            "dim": [32, 64] if small_dataset else [64, 128],
            "heads": [2, 4],
            "dropout": [0.1, 0.2, 0.3],
        }

    def fit(self, X, y, X_test, y_test):
       
       # Convert Pandas DataFrame to NumPy array before conversion
        X_train, y_train = check_X_y(X, y, accept_sparse=False)
        X_test, y_test = np.array(X_test), np.array(y_test)  # Ensure they are NumPy arrays

        # Convert to PyTorch tensors
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

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
            model = FTTransformer.make_default(
                    n_num_features= X.shape[1],  # Số lượng cột dữ liệu dạng số
                    cat_cardinalities=[],  # Độ lớn của mỗi feature dạng categorical
                    last_layer_query_idx=[-1],  # Chỉ số lớp cuối để dự đoán
                    d_out=len(set(y))  # Số đầu ra
                )
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(20):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    out = model(xb, None)  # or
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            y_preds = []
            y_true = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(self.device)
                    # FTTransformer expects both x_num and x_cat as inputs
                    out = model(xb, None)  # or provide an empty tensor for x_cat
                    y_pred = torch.argmax(out, dim=1).cpu().numpy()
                    y_preds.extend(y_pred)
                    y_true.extend(yb.numpy())
                    
            f1 = f1_score(y_true, y_preds, average="binary")
            acc = accuracy_score(y_true, y_preds)
            results.append({**param, "accuracy": acc, "f1_score": f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
        
        self.result_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
        self.model = best_model
        self.best_params_ = param_combinations[0] if best_model else None

    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)
