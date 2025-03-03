import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tabmini
import pickle
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

from tabmini.estimators.TabR import TabR
from tabmini.estimators.FTTransformer import FTTransformerClassifier 
from tabmini.estimators.ResNet import ResNet 
from tabmini.estimators.SAINT import SAINT 
from tabmini.estimators.RandomForest import RandomForest 
from tabmini.types import TabminiDataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="TabMini-Classification Options.")
    parser.add_argument(
        "--model",
        type=int,
        choices = [4, 5, 6, 8 , 11],
        default= 4,
        help="Type of model (4: Random Forest, 5: ResNet, 6: FTTransformer, 8: TabR, 11: SAINT)",
    )
    
    parser.add_argument(
        "--scale", 
        type= bool, 
        default= False, 
        help= "Apply Standard Scaler or not."
    )

    parser.add_argument(
        "--save_dir", type=str, default="result", help="Folder to save result."
    )

    return parser.parse_args()

def load_dataset():
    # Đường dẫn tệp pickle
    pickle_file = 'data/dataset.pkl'

    # Kiểm tra xem tệp pickle đã tồn tại hay chưa
    if os.path.exists(pickle_file):
        # Nếu tệp đã tồn tại, tải dataset từ tệp pickle
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
        print("Dataset đã được tải từ tệp pickle.")
    else:
        # Nếu tệp chưa tồn tại, tải dataset mới và lưu vào tệp pickle
        dataset = tabmini.load_dataset(reduced=False)
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset, f)
        print("Dataset đã được tải và lưu vào tệp pickle.")
    
    return dataset

def main(args):
    working_directory = Path.cwd() / args.save_dir
    working_directory.mkdir(parents=True, exist_ok=True)

    #1. Tải dataset và lưu vào data/dataset.pkl
    # Đường dẫn tệp pickle
    pickle_file = 'tabmini/data/dataset.pkl'
    # Kiểm tra xem tệp pickle đã tồn tại hay chưa
    if os.path.exists(pickle_file):
        # Nếu tệp đã tồn tại, tải dataset từ tệp pickle
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
        print("Dataset đã được tải từ tệp pickle.")
    else:
        # Nếu tệp chưa tồn tại, tải dataset mới và lưu vào tệp pickle
        dataset: TabminiDataset = tabmini.load_dataset(reduced=False)
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset, f)
        print("Dataset đã được tải và lưu vào tệp pickle.")

    #2. Tiền xử lý data và train
    dataset_list = list(dataset.keys())
    for dt_name in dataset_list:
        X, y = dataset[dt_name]
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        if 2 in y.values:
            y = (y == 2).astype(int)
        num_records = len(X)

        # preprocessing data 
        if args.scale: 
            X = tabmini.normalize_data(X)

        X_train, X_test, y_train, y_test = tabmini.split_train_test(X, y)

        # train and predict        
        if args.model == 4:
            model = RandomForest(small_dataset=True)
        elif args.model == 5:
            model = ResNet(input_dim, num_classes) 
        elif args.model == 6:
            model = FTTransformerClassifier(small_dataset= True) 
        elif args.model == 8: 
            model = TabR(small_dataset=True)
        elif args.model == 11: 
            model = SAINT(small_dataset = True)
        # else:
        #    model_RF = RandomForest(small_dataset=True)
        #    model_RF.fit(X_train, y_train, X_test, y_test)
        #    filename = os.path.join("result", f"{dt_name}_{num_records}.csv")
        #    model_RF.save_results(filename=filename)

        #    model_RestNet = ResNet(small_dataset= True) 
        #    model_RestNet.fit(X_train, y_train, X_test, y_test)
        #    filename = os.path.join("result", f"{dt_name}_{num_records}.csv")
        #    model_RestNet.save_results(filename=filename)

        #    model_FT = FTTransformer(small_dataset= True) 
        #    model_FT.fit(X_train, y_train, X_test, y_test)
        #    filename = os.path.join("result", f"{dt_name}_{num_records}.csv")
        #    model_FT.save_results(filename=filename)

        #    model_TabR = TabR(small_dataset=True)
        #    model_TabR.fit(X_train, y_train, X_test, y_test)
        #    filename = os.path.join("result", f"{dt_name}_{num_records}.csv")
        #    model_TabR.save_results(filename=filename)

        #    model_SAINT = SAINT(small_dataset = True)
        #    model_SAINT.fit(X_train, y_train, X_test, y_test)
        #    filename = os.path.join("result", f"{dt_name}_{num_records}.csv")
        #    model_SAINT.save_results(filename=filename)

        model.fit(X_train, y_train, X_test, y_test)
        filename = os.path.join(working_directory, f"{dt_name}_{num_records}.csv")
        model.save_results(filename=filename)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)


# # Ensure target labels are binary (0 and 1)
# def fix_target_labels(y):
#     y = np.where(y == 2, 1, y)  # Replace label 2 with 1
#     return y

# # Define TabTransformer Model
# class TabTransformer(nn.Module):
#     def __init__(self, input_dim, num_classes, hidden_dim=128, num_heads=8, num_layers=4):
#         super(TabTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, hidden_dim)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
#             num_layers=num_layers
#         )
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.transformer(x)
#         x = self.fc(x)
#         return self.softmax(x)

# # Training Function for TabTransformer
# def train_tabtransformer(X_train, y_train, X_test):
#     input_dim = X_train.shape[1]
#     model = TabTransformer(input_dim=input_dim, num_classes=2)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Convert data to PyTorch tensors
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#     # Training loop
#     for epoch in range(10):  # 10 epochs (can be increased for better performance)
#         model.train()
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
    
#     # Inference
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X_test_tensor)
#         y_pred = predictions.argmax(dim=1).numpy()

#     return y_pred

# # 1. Random Forest
# def train_random_forest(X_train_scaled, y_train, X_test_scaled):
#     model = RandomForestClassifier(n_estimators=100)
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     return y_pred
# # 2. ResNet
# config_resnet = {
#     # Model Information
#     "model": "resnet",
    
#     # Model Architecture
#     "activation": "relu",
#     "d": 466,
#     "d_embedding": 0,
#     "d_hidden_factor": 3.39553393975761,
#     "hidden_dropout": 0.152157876156,
#     "n_layers": 7,
#     "normalization": "batchnorm",
#     "residual_dropout": 0.0,
    
#     # Training Info
#     "optim": "AdamW",
#     "lr": 2.055922372891508e-05,
#     "weight_decay": 0.0,
#     "batchsize": 512,
#     "epochs": 10000000,
#     "fold": 0,
#     "count": 0
# }
# def train_res_net(X_train, y_train, X_test):
#     input_dim = X_train.shape[1]
#     criterion = nn.CrossEntropyLoss()
#     model = ResNet(
#                     d_numerical= input_dim,
#                     categories = None,

#                     # ModelA Architecture
#                     activation = "relu",
#                     d = int(config_resnet["d"]),
#                     d_embedding = int(config_resnet["d_embedding"]),
#                     d_hidden_factor = float(config_resnet["d_hidden_factor"]), 
#                     hidden_dropout = float(config_resnet["hidden_dropout"]),
#                     n_layers = int(config_resnet["n_layers"]),
#                     normalization = config_resnet["normalization"],
#                     residual_dropout = float(config_resnet["residual_dropout"]),

#                     # default_Setting
#                     d_out = input_dim
#         )
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     optimizer = optim.Adam(
#             model.parameters(),
#             lr = float(config_resnet["lr"]),
#             weight_decay = float(config_resnet["weight_decay"]),
#             eps = 1e-8
#         )
#     for epoch in range(50):  # 10 epochs (can be increased for better performance)
#         model.train()
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X, x_cat = None)
#             loss = criterion(outputs.squeeze(1), batch_y)
#             loss.backward()
#             optimizer.step()
    
#     # Inference
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X_test_tensor, x_cat = None)
#         y_pred = predictions.argmax(dim=1).numpy()

#     return y_pred

# # 3. FT-Transformer
# config_ft = {
#     # Model Information
#     "model": "ft-transformer",
    
#     # Model Architecture
#     "activation": "reglu",
#     "attention_dropout": 0.2,
#     "d_ffn_factor": 1.3333333333333,
#     "d_token": 192,
#     "ffn_dropout": 0.1,
#     "initialization": "kaiming",
#     "n_heads": 8,
#     "n_layers": 3,
#     "prenormalization": True,
#     "residual_dropout": 0.0,
    
#     # Training Info
#     "optim": "AdamW",
#     "lr": 1e-4,
#     "weight_decay": 1e-05,
#     "batchsize": 512,
#     "epochs": 10000000,
#     "fold": 0,
#     "count": 0,
    
#     # Additional Parameters
#     "kv_compression": 0,
#     "kv_compression_sharing": 0
# }

# def train_tf_transformer(X_train, y_train, X_test):
#     input_dim = X_train.shape[1]
#     criterion = nn.CrossEntropyLoss()
#     model = Transformer(d_numerical = input_dim,
#                         categories = None,
#                         # Model Architecture
#                         n_layers = int(config_ft["n_layers"]),
#                         n_heads = int(config_ft["n_heads"]),
#                         d_token = int(config_ft["d_token"]),
#                         d_ffn_factor = float(config_ft["d_ffn_factor"]),
#                         attention_dropout = float(config_ft["attention_dropout"]),
#                         ffn_dropout = float(config_ft["attention_dropout"]),
#                         residual_dropout = float(config_ft["residual_dropout"]),
#                         activation = config_ft["activation"],
#                         prenormalization = True,
#                         initialization = config_ft["initialization"],
                        
#                         # default_Setting
#                         token_bias = True,
#                         kv_compression = None if int(config_ft["kv_compression"]) == 0 else int(config_ft["kv_compression"]),
#                         kv_compression_sharing= None if int(config_ft["kv_compression"]) == 0 else float(config_ft["kv_compression"]),
#                         d_out = input_dim
#                     )
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     optimizer = optim.Adam(
#             model.parameters(),
#             lr = float(config_ft["lr"]),
#             weight_decay = float(config_ft["weight_decay"]),
#             eps = 1e-8
#         )
#     for epoch in range(10):  # 10 epochs (can be increased for better performance)
#         model.train()
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X, x_cat = None)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
    
#     # Inference
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X_test_tensor, x_cat = None)
#         y_pred = predictions.argmax(dim=1).numpy()

#     return y_pred
# # 4. TabR
# def train_tabr(X_train, y_train, X_test):
#     input_dim = X_train.shape[1]
#     model = TabR(
#     n_num_features=input_dim,  # 5 đặc trưng số
#     n_bin_features=2,  # 3 đặc trưng nhị phân
#     cat_cardinalities=[10, 10, 10, 10],  # Các đặc trưng phân loại có 10 giá trị duy nhất
#     n_classes=None,  # Hồi quy (nếu phân loại, thì n_classes sẽ > 2)
#     d_main=64,
#     d_multiplier=1.5,
#     encoder_n_blocks=2,
#     predictor_n_blocks=2,
#     mixer_normalization='LayerNorm',
#     context_dropout=0.2,
#     dropout0=0.1,
#     dropout1=0.1,
#     normalization='LayerNorm',
#     activation='ReLU',
#     memory_efficient=False,
# )

# # 5. SAINT (Using TabTransformer)
# def train_saint(X_train_scaled, y_train, X_test_scaled):
#     return train_tabtransformer(X_train_scaled, y_train, X_test_scaled)

# # Model Evaluation
# def evaluate_model(y_true, y_pred):
#     accuracy = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
    
#     # Check if there are at least two classes in y_true
#     if len(np.unique(y_true)) == 2:
#         auc = roc_auc_score(y_true, y_pred)
#     else:
#         auc = None  # If only one class, AUC can't be computed
    
#     return accuracy, f1, auc

# # Create DataFrame to store results for each model
# lightgbm_results = []
# rf_results = []
# resn_results = []
# tabtransformer_results = []
# saint_results = []
# ft_results = []

# # Iterate through all datasets and train for each
# for dataset_name, (X, y) in dataset.items():
#     print(f"\nTraining with dataset: {dataset_name}")
    
#     # Ensure target labels are binary (0 and 1)
#     y = fix_target_labels(y)
    
#     # Split dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Data Preprocessing
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     #TF-Transformer
#     ft_pred = train_tf_transformer(X_train_scaled, y_train, X_test_scaled)
#     ft_score = evaluate_model(y_test, ft_pred)
#     ft_results.append([dataset_name, *ft_score])
#     #ResNet
#     resn_pred = train_res_net(X_train_scaled, y_train, X_test_scaled)
#     resn_score = evaluate_model(y_test, resn_pred)
#     resn_results.append([dataset_name, *resn_score])
#     # Random Forest
#     # rf_pred = train_random_forest(X_train_scaled, y_train, X_test_scaled)
#     # rf_score = evaluate_model(y_test, rf_pred)
#     # rf_results.append([dataset_name, *rf_score])
    
#     # SAINT
#     # saint_pred = train_saint(X_train_scaled, y_train, X_test_scaled)
#     # saint_score = evaluate_model(y_test, saint_pred)
#     # saint_results.append([dataset_name, *saint_score])

# # # Convert results into DataFrame
# # rf_results_df = pd.DataFrame(rf_results, columns=["Dataset", "Accuracy", "F1", "AUC"])
# # saint_results_df = pd.DataFrame(saint_results, columns=["Dataset", "Accuracy", "F1", "AUC"])
# ft_results_df = pd.DataFrame(ft_results, columns=["Dataset", "Accuracy", "F1", "AUC"])
# resn_results_df = pd.DataFrame(resn_results, columns=["Dataset", "Accuracy", "F1", "AUC"])

# # # Save results to CSV files
# # rf_results_df.to_csv("rf_results.csv", index=False)
# # saint_results_df.to_csv("saint_results.csv", index=False)
# ft_results_df.to_csv("ft_result.csv", index=False)
# resn_results_df.to_csv("resn_result.csv", index=False)