import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from script.metrics import ndcg_at_k_minmax
from sklearn.preprocessing import StandardScaler
from deep_model import Classifier 
from collections import defaultdict
from sklearn.model_selection import KFold
import glob
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def predict_with_pretrained_model(pre_train_path, x_test, 
                                  hidden_size=128, dropout_rate=0.5):

    """
    Predict outputs using pre-trained model weights (without fine-tuning).

    Args:
        pre_train_path (str): Directory path containing pre-trained weights.
        x_test (array-like): Test dataset.
        hidden_size (int): Hidden layer size of the classifier.
        dropout_rate (float): Dropout rate for the classifier.

    Returns:
        pd.DataFrame: Predictions from all folds (columns = folds).
    """

    result = pd.DataFrame()
    test_input = torch.tensor(x_test, dtype=torch.float32).to(device)

    # Loop through 10 folds
    for fold in range(10):
        model = Classifier(hidden_size, dropout_rate).to(device)

        # Load pre-trained weights
        weight_path = f"{pre_train_path}/pretrained_weight_{fold}.pth"
        model.load_state_dict(_load_state_dict(weight_path))

        model.eval()

        # Inference on test data
        with torch.no_grad():
            preds = model(test_input).cpu().numpy()
        result[fold] = preds

    return result


def predict_with_enzyme_specific_prediction(test_enzyme, pre_train_path,
                             x_train, y_train, x_test, 
                             hidden_size=128, dropout_rate=0,
                             ft_epochs=50, learning_rate=1e-5, fold_num=10):

    """
    Fine-tune a pre-trained model with new training data, 
    then predict on test data.

    Args:
        test_enzyme (str): Name of the enzyme (for logging/saving if needed).
        pre_train_path (str): Directory path containing pre-trained weights.
        x_train, y_train (array-like): Training dataset and labels.
        x_test (array-like): Test dataset.
        hidden_size (int): Hidden layer size of the classifier.
        dropout_rate (float): Dropout rate for the classifier.
        ft_epochs (int): Number of epochs for fine-tuning.
        learning_rate (float): Learning rate for fine-tuning.
        fold_num (int): Number of folds.

    Returns:
        pd.DataFrame: Predictions from all folds (columns = folds).
    """

    result = pd.DataFrame()

    scaler_x = StandardScaler().fit(x_train)
    x_train_scaled = scaler_x.transform(x_train)
    x_test_scaled = scaler_x.transform(x_test) 

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    for fold in range(fold_num):
        model = Classifier(hidden_size, dropout_rate).to(device)

        # pre-trained weight load
        weight_path = f"{pre_train_path}/pretrained_weight_{fold}.pth"
        model.load_state_dict(_load_state_dict(weight_path))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Freeze first conv layer during fine-tuning
        for p in model.conv1.parameters():
            p.requires_grad = False

        # DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, drop_last=True)
        torch.set_grad_enabled(True)

        # Fine-tuning
        for epoch in range(ft_epochs):
            model.train()
            for batch_inputs, batch_labels in train_loader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                optimizer.zero_grad()

                batch_outputs = model(batch_inputs)
                loss = criterion(batch_outputs, batch_labels)

                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = model(x_test_tensor).cpu().numpy()
        result[fold] = preds

    return result

def train_model(test_enzyme, x_train, y_train, x_test, y_test, weight_dir,version,
                hidden_size=128, dropout_rate=0.5, num_epochs=700,
                learning_rate=0.01, fold_num=10):
    """
    Train and evaluate a regression model with K-Fold cross-validation.

    Args:
        test_enzyme (str): Name of the enzyme (used for saving weights).
        x_train, y_train (array-like): Training data and labels.
        x_test, y_test (array-like): Test data and labels.
        weight_dir (str): Directory where model weights are saved.
        version (str): version of train set. 
        hidden_size (int): Hidden layer size of the classifier.
        num_epochs (int): Number of epochs for training each fold.
        dropout_rate (float): Dropout rate for the classifier.
        learning_rate (float): Initial learning rate.
        fold_num (int): Number of folds.

    Returns:
        pd.DataFrame: Predictions from all folds (columns = folds).
    """

    result = pd.DataFrame()
    criterion = nn.MSELoss()
    starting_point = 0

    # Find existing trained folds (resume from the last fold if exists)
    file_pattern= f"{weight_dir}/{version}_*_fcn_weight.pth"
    files = glob.glob(file_pattern)
    nums = []
    for file in files:
        match = re.search(rf"{version}_(\d+)_fcn_weight\.pth", os.path.basename(file))
        if match:
            nums.append(match.group(1))
    if nums:
        starting_point = max(map(int, nums))

    fold=starting_point

    # Remove NaN values from training data
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    mask = ~np.isnan(x_train).any(axis=1)
    x_train_valid = x_train[mask]
    y_train_valid = y_train[mask]

    # Start training with K-Fold CV
    for random_seed in range(starting_point, fold_num):
        kf=KFold(n_splits=5,shuffle=True,random_state=random_seed)
        train_index, val_index = next(kf.split(x_train_valid))

        scaler_x = StandardScaler().fit(x_train_valid)
        x_train_scaled = scaler_x.transform(x_train_valid)
        x_test_scaled = scaler_x.transform(x_test) 

        # Split into train/val sets
        x_train_fold = torch.tensor(x_train_scaled[train_index], dtype=torch.float32).to(device)
        y_train_fold = torch.tensor(y_train_valid[train_index], dtype=torch.float32).to(device)
        x_val_fold = torch.tensor(x_train_scaled[val_index], dtype=torch.float32).to(device)
        y_val_fold = torch.tensor(y_train_valid[val_index], dtype=torch.float32).to(device)

        train_dataset = TensorDataset(x_train_fold, y_train_fold)
        val_dataset = TensorDataset(x_val_fold, y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=True)

        # Initialize model and optimizer        
        model= Classifier(hidden_size, dropout_rate).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
        torch.set_grad_enabled(True)

        # Training loop
        high_val_ndcg=0
        for epoch in range(num_epochs):
            model.train()
            total_loss,total_ndcg=0,0

            # Training step
            for batch_inputs, batch_labels in train_loader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                batch_outputs = model(batch_inputs)
                loss = criterion(batch_outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                batch_ndcg = ndcg_at_k_minmax(batch_outputs.tolist(), batch_labels.tolist(), len(batch_labels))
                total_ndcg += batch_ndcg

            scheduler.step()

            train_loss = total_loss/len(train_loader)
            train_ndcg = total_ndcg/len(train_loader)

            # Validation step
            model.eval()
            val_loss, val_ndcg = 0, 0
            with torch.no_grad():
               for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_labels)
                    val_loss += loss.item()
                    val_ndcg += ndcg_at_k_minmax(val_outputs.tolist(), val_labels.tolist(), len(val_labels))

            val_loss /= len(val_loader)
            val_ndcg/=len(val_loader)

            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"[Fold {fold}][Epoch {epoch+1}] "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train NDCG: {train_ndcg:.4f}, Val NDCG: {val_ndcg:.4f}")

            # Save the best model based on validation NDCG
            if val_ndcg>high_val_ndcg:
                high_val_ndcg=val_ndcg
                print("######################################################")
                print(f"epoch {epoch}, best val_ndcg {high_val_ndcg:.4f}")
                torch.save(model.state_dict(),f'{weight_dir}/{version}_{fold}_fcn_weight.pth')

        print(f"Fold {fold} - Validation Accuracy: {val_loss:.4f}")
        fold=fold+1

    # Final evaluation on test set
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    for fold in range(fold_num):
        model= Classifier(hidden_size, dropout_rate).to(device)
        model.load_state_dict(torch.load(
            f"{weight_dir}/{version}_{fold}_fcn_weight.pth",
        ))
        model.eval()
        
        with torch.no_grad():
            preds =model(x_test_tensor)
        result[fold] = preds.cpu().numpy()

    return result


