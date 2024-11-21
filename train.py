# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

# %%
targetFeatures=['tavg','tmin','tmax','prcp','snow','wspd','pres']
input_size = len(targetFeatures) 
hidden_size = 64      # Number of hidden units in LSTM
num_layers = 2        # Number of LSTM layers
output_size =  len(targetFeatures) 
seq_length = 30       # Sequence length (number of past days to consider)


# %%
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # LSTM Layer forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size)
        
        # Check if lstm_out has 3 dimensions or just 2
        if lstm_out.dim() == 3:
            # If lstm_out has 3 dimensions, we access the last time step
            out = lstm_out[:, -1, :]  # Get the output from the last time step for each batch
        else:
            # If lstm_out has 2 dimensions (batch_size, hidden_size), just use it
            out = lstm_out
        
        # Pass through the fully connected layer
        out = self.fc(out)
        return out



# %%
df=pd.read_csv('newYork.csv')
df.drop(['wpgt','tsun','wdir'],axis=1,inplace=True)
df.fillna(0,inplace=True)
df.replace([float('inf'), -float('inf')], 0, inplace=True)
print(df.isnull().sum())  # To check for any NaN values
print((df == float('inf')).sum())

# %%

from sklearn.preprocessing import *
# Assuming your data is in the form of pandas DataFrame `df` and your features and targets are already separated
# Let's say the columns are 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'pres'

# Features and targets
features = df[targetFeatures].values
targets = df[targetFeatures].values  # Same columns for now, could be adjusted

# Split the data into training and validation sets (e.g., 80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_train_reshaped = X_train_tensor.view(-1, X_train_tensor.size(-1))
X_val_reshaped = X_val_tensor.view(-1, X_val_tensor.size(-1))

# 2. Initialize MinMaxScaler and apply it
scaler = QuantileTransformer(output_distribution='uniform')
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_val_scaled = scaler.transform(X_val_reshaped)

# 3. Reshape back to 3D (batch_size, sequence_length, features)
X_train_scaled = torch.tensor(X_train_scaled).view(X_train_tensor.shape)
X_val_scaled = torch.tensor(X_val_scaled).view(X_val_tensor.shape)

# Convert scaled data back to tensors
X_train_scaled = X_train_scaled.float()
X_val_scaled = X_val_scaled.float()

# 4. Create TensorDataset and DataLoader
y_train_tensor = y_train_tensor.float()  # Make sure target tensors are float as well
y_val_tensor = y_val_tensor.float()

train_dataset = TensorDataset(X_train_scaled, y_train_tensor)
val_dataset = TensorDataset(X_val_scaled, y_val_tensor)

batch_size = 32  # Set your batch size as needed

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# %%
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Assuming you're using a device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# Define loss and optimizer
criterion = nn.L1Loss()
# optimizer = optim.RMSprop(model.parameters(), lr=1, alpha=0.9, weight_decay=0.001)
from torch_optimizer import Lookahead
base_optimizer =optim.RMSprop(model.parameters(), lr=1, alpha=0.1, weight_decay=0.0001)
optimizer = Lookahead(base_optimizer, k=15, alpha=1)


# Reduce learning rate
import torch.optim.lr_scheduler as lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# Enable mixed precision if you're using GPU
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# Sample training loop with optimizations
num_epochs = 20 
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (seq, target) in enumerate(train_loader):
        seq, target = seq.to(device), target.to(device)

        # Mixed precision forward pass
        with autocast():
            output = model(seq)
            loss = criterion(output, target)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights with gradient scaling
        if (i + 1) % 4 == 0:  # Gradient accumulation every 4 mini-batches
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()

        # Print training loss every 100 iterations (optional)
        if (i + 1) % 100 == 0:
            avg_train_loss = train_loss / (i + 1)

    # Average training loss for the entire epoch
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}")
    scheduler.step()


# %%

from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
import numpy as np
import torch

# Define the evaluation function with additional metrics
def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    val_loss = 0.0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for seq, target in val_loader:
            # Move data to the device (GPU or CPU)
            seq, target = seq.to(device), target.to(device)

            # Forward pass
            output = model(seq)

            # Calculate the loss
            loss = criterion(output, target)
            val_loss += loss.item()

            # Collect predictions and true labels for metrics
            all_preds.append(output.cpu().numpy())  # Move to CPU and convert to numpy
            all_labels.append(target.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate additional metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)  # Mean Absolute Error
    r2 = r2_score(all_labels, all_preds)  # R-squared score
    evs = explained_variance_score(all_labels, all_preds)  # Explained Variance Score

    avg_val_loss = val_loss / len(val_loader)  # Average validation loss

    # Print the evaluation results
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")

    return avg_val_loss, rmse, mae, r2, evs


# Example usage:
# Assuming `val_loader` is your validation DataLoader
# `model` is the trained model, `criterion` is the loss function (MSELoss)
# `device` is your computation device (GPU or CPU)
val_loss, rmse, mae, r2, evs = evaluate_model(model, val_loader, criterion, device)



# %%
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')


