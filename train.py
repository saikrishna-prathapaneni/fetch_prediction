
import argparse
import joblib
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from model import LinearRegressionModel
from torch.utils.data import DataLoader, TensorDataset



def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure the date column is in datetime format
    return data



# Receipt_Count','Month','Day','Lag_1', 'Rolling_Mean_7' used for the model

# Feature engineering
def feature_engineering(data):
    
    # Extracting time features from the data
    #data['Year'] = data['Date'].dt.year # Not significant in this case
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Lag_1'] = data['Receipt_Count'].shift(1)

    # Rolling window features (e.g., rolling average of the past 7 days)
    data['Rolling_Mean_7'] = data['Receipt_Count'].rolling(window=7).mean()

    # Drop rows with NaN values which are a result of lagged features
    data = data.dropna()

    return data

def process_file(file_path):
    data = load_data(file_path)
    data = feature_engineering(data)
    return data

def validate(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            inputs = inputs.view(-1, args.input_size)
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item()
        average_loss = total_loss / len(test_loader)
    return average_loss

def train_one_epoch(args, model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # Forward pass
        inputs = inputs.view(-1, args.input_size)
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))

        # Apply L1 regularization
        if args.include_l1reg:
            l1_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l1_reg = l1_reg + torch.norm(param, 1)
            loss = loss + args.l1_lambda * l1_reg

       

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get speed of the conveyor")
    
    parser.add_argument("--datafile",
                        type=str, default="data_daily.csv", help ="location to the data file")
    
    parser.add_argument("--include_l2reg",
                        type=bool, default = False, help ="use l2 regularisation while training")
    
    parser.add_argument("--include_l1reg",
                        type=bool, default = False, help ="use l1 regularisation while training")
    
    parser.add_argument("--l1_lambda",
                        type=float, default = 0.001, help ="value of lambda in l1")
    
    parser.add_argument("--l2_lambda",
                        type=float, default = 0.003, help =" value of lambda in l2")
    
    parser.add_argument("--lr",
                        type=float, default = 0.01, help ="learning rate for training the model")
    
    parser.add_argument("--batch_size",
                        type=int, default = 8, help ="batch size for training the model")
    
    parser.add_argument("--epochs",
                        type=int, default = 20, help ="number of epochs for training the model")


    
    args = parser.parse_args()
 
    
    data = process_file(args.datafile)

    df_sub = data.copy()

    # features considered after experimentation
    data_f = data[['Receipt_Count','Month','Day','Lag_1', 'Rolling_Mean_7']]
    X = data_f.drop('Receipt_Count', axis=1).values
    y = data_f['Receipt_Count'].values.astype(float)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    # Scaling data
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_scaler = MinMaxScaler()

    # Reshape y_train and y_test to be a 2D array as MinMaxScaler expects
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # save the x scaler
    joblib.dump(scaler, 'assets/x_scaler.pkl')

    # Save the y scaler
    joblib.dump(y_scaler, 'assets/y_scaler.pkl')

    # Convert arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    args.input_size = X_train.shape[1]
    model = LinearRegressionModel(args.input_size)

  
    # Loss and Optimizer
    criterion = nn.MSELoss()
     
    # Apply L2 regularization is already applied through weight_decay in optimizer
    if args.include_l2reg:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.l2_lambda)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
   
    # Training the Model
    num_epochs = args.epochs
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(args, model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)

        test_loss = validate(model, test_loader, criterion)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, "model_weights/linear_model.pt")
    
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()