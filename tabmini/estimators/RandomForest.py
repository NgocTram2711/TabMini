import numpy as np
import pandas as pd

from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        time_limit: int = 3600,
        seed: int = 42,
        kwargs: dict = {},
        small_dataset: bool = False,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.kwargs = kwargs
        self.result_df = None

        # Grid các tham số đã được cập nhật
        self.param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_leaf_nodes": [10, 20, 30]
        }

    def fit(self, X, y, X_test, y_test) -> "RandomForest":
        X_train, y_train = check_X_y(X, y, accept_sparse=True)

        results = []
        best_f1 = -1
        best_model = None

        # Tạo ra tất cả các tổ hợp tham số từ grid
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(
                -1, len(self.param_grid.keys())
            )
        ]
        for param in param_combinations:
            # Ép kiểu số nguyên cho các tham số cần thiết
            param["n_estimators"] = int(param["n_estimators"])
            param["max_depth"] = int(param["max_depth"])
            param["min_samples_split"] = int(param["min_samples_split"])
            param["min_samples_leaf"] = int(param["min_samples_leaf"])
            param["max_leaf_nodes"] = int(param["max_leaf_nodes"])

            current_model = RandomForestClassifier(**param, random_state=self.seed)
            current_model.fit(X_train, y_train)

            # Dự đoán và đánh giá
            # Chuyển X_test thành numpy array nếu nó là DataFrame để tránh cảnh báo về feature names
            if hasattr(X_test, "columns"):
                X_test_mod = X_test.values
            else:
                X_test_mod = X_test

            y_pred = current_model.predict(X_test_mod)
            f1 = f1_score(y_test, y_pred, average="binary")
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="binary")
            recall = recall_score(y_test, y_pred, average="binary")
            auc = roc_auc_score(y_test, current_model.predict_proba(X_test_mod)[:, 1])

            results.append({
                **param,
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "auc_roc": auc
            })

            # Cập nhật model tốt nhất dựa trên f1_score
            if f1 > best_f1:
                best_f1 = f1
                best_model = current_model

        self.result_df = pd.DataFrame(results).sort_values(
            by="f1_score", ascending=False
        )
        self.model = best_model
        self.best_params_ = best_model.get_params() if best_model else None

    def predict_proba(self, X):
        check_is_fitted(self)
        # Chuyển đổi X thành numpy array nếu là DataFrame
        if hasattr(X, "columns"):
            X = X.values
        X = check_array(X, accept_sparse=True)
        probability_positive_class = self.model.predict_proba(X)[:, 1]
        probability_positive_class_scaled = (
            probability_positive_class - probability_positive_class.min()
        ) / (
            probability_positive_class.max() - probability_positive_class.min() + 1e-10
        )
        return np.vstack(
            [1 - probability_positive_class_scaled, probability_positive_class_scaled]
        ).T

    def decision_function(self, X):
        # Dùng predict_proba đã được cập nhật nên không cần chuyển đổi thêm ở đây
        proba = self.predict_proba(X)
        decision = np.log((proba[:, 1] + 1e-10) / (proba[:, 0] + 1e-10))
        return decision

    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)
