import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from collections import defaultdict

from deep_model import Classifier  # 모델 정의

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def predict_with_pretrained_model(pre_train_path, x_test, 
                                  hidden_size=128, dropout_rate=0.5):
    result = pd.DataFrame()
    test_input = torch.tensor(x_test, dtype=torch.float32).to(device)

    for fold in range(10):
        model = Classifier(hidden_size, dropout_rate).to(device)
        weight_path = f"{pre_train_path}/pretrained_weight_{fold}.pth"
        model.load_state_dict(_load_state_dict(weight_path))
        model.eval()

        with torch.no_grad():
            preds = model(test_input).cpu().numpy()
        result[fold] = preds

    return result


def predict_with_fine_tuning(test_enzyme,pre_train_path,
                             x_train, y_train, x_test,result_dir, 
                             hidden_size=128, dropout_rate=0, 
                             fittransform=False, ft_epochs=50,
                             learning_rate=1e-5):
                    
    result = pd.DataFrame()
    test_input = torch.tensor(x_test, dtype=torch.float32).to(device)

    train_input = torch.tensor(x_train, dtype=torch.float32).to(device)
    train_target = torch.tensor(y_train, dtype=torch.float32).to(device)

    for fold in range(10):
        model = Classifier(hidden_size, dropout_rate).to(device)

        # pre-trained weight load
        if pre_train_path != "none":
            weight_path = f"{pre_train_path}/pretrained_weight_{fold}.pth"
            model.load_state_dict(_load_state_dict(weight_path))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # layer freeze
        for p in model.conv1.parameters():
            p.requires_grad = False

        # DataLoader
        train_loader = DataLoader(
            TensorDataset(train_input, train_target),
            batch_size=24, shuffle=True, drop_last=True
        )

        torch.set_grad_enabled(True)
        # Fine-tuning
        for epoch in range(ft_epochs):
            for inputs, labels in train_loader:
                model.train()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = model(test_input).cpu().numpy()
        result[fold] = preds


    return result
