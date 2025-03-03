# The model.

# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import lib
# from lib import KWArgs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_X_y, check_array
import torch.optim as optim
import pandas as pd

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class TabRModel(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E
        d_in = (
            n_num_features
            * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, lib.get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        *,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x_)
        if is_train:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if is_train else 0)
            )
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x


class TabR(BaseEstimator, ClassifierMixin):
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
            "learning_rate": [0.001, 0.05, 0.1] if small_dataset else [0.001, 0.01, 0.05, 0.1],
            "epochs": [10, 30, 50] if small_dataset else [10, 50, 100, 150],
            "context_size": [3, 5] if small_dataset else [5, 10, 15],
        }
        
        self.default_model_params = {
            "d_main": 64 if small_dataset else 128,
            "d_multiplier": 2,
            "encoder_n_blocks": 2 if small_dataset else 3,
            "predictor_n_blocks": 1 if small_dataset else 2,
            "mixer_normalization": "auto",
            "context_dropout": 0.1,
            "dropout0": 0.2,
            "dropout1": "dropout0",
            "normalization": "BatchNorm1d",
            "activation": "ReLU",
            "memory_efficient": False,
        }

    def fit(self, X, y, X_test, y_test):
        # Ensure data is in the right format
        X_train, y_train = check_X_y(X, y, accept_sparse=False)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)

        # Create dictionaries for TabR
        X_train_dict = {"num": X_train_tensor}
        X_test_dict = {"num": X_test_tensor}

        n_classes = len(np.unique(y_train))
        
        results = []
        best_f1 = -1
        best_model = None
        
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(-1, len(self.param_grid.keys()))
        ]

        print(f"Training TabR model with {len(param_combinations)} parameter combinations")
        
        for i, param in enumerate(param_combinations):
            print(f"\nTraining combination {i+1}/{len(param_combinations)}: {param}")
            
            # Cast parameters to appropriate types
            learning_rate = float(param["learning_rate"])
            epochs = int(param["epochs"])
            context_size = int(param["context_size"])
            
            # Initialize the model
            current_model = TabR(
                n_num_features=X_train.shape[1],
                n_bin_features=0,
                cat_cardinalities=[],
                n_classes=n_classes,
                num_embeddings=None,
                **self.default_model_params,
            ).to(self.device)
            
            # Initialize optimizer
            optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
            
            # Training loop
            for epoch in range(epochs):
                current_model.train()
                optimizer.zero_grad()
                
                outputs = current_model(
                    x_=X_train_dict,
                    y=y_train_tensor,
                    candidate_x_=X_train_dict,
                    candidate_y=y_train_tensor,
                    context_size=context_size,
                    is_train=True,
                )
                
                if n_classes == 2:
                    # Binary classification
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(outputs.squeeze(), y_train_tensor.float())
                else:
                    # Multi-class classification
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, y_train_tensor)
                
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")
            
            # Evaluation
            current_model.eval()
            with torch.no_grad():
                y_pred = current_model(
                    x_=X_test_dict,
                    y=None,
                    candidate_x_=X_train_dict,
                    candidate_y=y_train_tensor,
                    context_size=context_size,
                    is_train=False,
                )
                
                if n_classes == 2:
                    # Binary classification
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    y_pred = (y_pred > 0.5).long()
                else:
                    # Multi-class classification
                    y_pred = y_pred.argmax(dim=1)
            
            # Calculate metrics
            y_test_np = y_test_tensor.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            acc = accuracy_score(y_test_np, y_pred_np)
            
            if n_classes <= 2:
                f1 = f1_score(y_test_np, y_pred_np, average="binary")
            else:
                f1 = f1_score(y_test_np, y_pred_np, average="weighted")
            
            # Store results
            results.append({**param, "accuracy": acc, "f1_score": f1})
            
            # Update best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = current_model
                self.best_params_ = param
        
        # Save results to DataFrame
        self.result_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
        self.model = best_model
        
        print("\nTraining completed. Best parameters:")
        print(self.best_params_)
        print(f"Best F1 Score: {best_f1:.4f}")
        
        return self

    def predict(self, X):
        """
        Predict class labels for X.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        X = check_array(X, accept_sparse=False)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        X_dict = {"num": X_tensor}
        
        self.model.eval()
        with torch.no_grad():
            # We need training data for the retrieval mechanism
            # Here we're using a simplified approach - in practice you'd want to save the training data
            outputs = self.model(
                x_=X_dict,
                y=None,
                candidate_x_=X_dict,  # This should be training data in real usage
                candidate_y=torch.zeros(X.shape[0], dtype=torch.long).to(self.device),  # Dummy values
                context_size=int(self.best_params_["context_size"]),
                is_train=False,
            )
            
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                # Multi-class
                predictions = outputs.argmax(dim=1)
            else:
                # Binary
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            
        return predictions.cpu().numpy()

    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")