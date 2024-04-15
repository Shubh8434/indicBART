# -*- coding: utf-8 -*-
"""pointer.ipynb


"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

new_df = pd.read_pickle("merged_embedding_summary.pkl")
new_df

new_df["Merged Embedding"][0].shape

import torch
import torch.nn as nn

class ModifiedLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModifiedLSTMModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x.to(torch.float32))
        output, _ = self.gru1(x)
        output, _ = self.gru2(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc4(output)
        return torch.sigmoid(output)

from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, embed,label):
        self.embeddings = embed.tolist()
        self.labels = label.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        embedding = torch.tensor(self.embeddings[index])
        label = torch.tensor(self.labels[index])
        return embedding, label

def create_data_loaders(dataframe, batch_size, test_size=0.1):
    train_df, test_df = train_test_split(dataframe, test_size=test_size, shuffle = False)
    X_train,y_train = train_df['Merged Embedding'],train_df['Label']
    X_test,y_test = test_df['Merged Embedding'],test_df['Label']
    print(X_train.shape)
    oversampler = RandomOverSampler(random_state=42)

    # Resample the training data
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train.values.reshape(-1, 1), y_train.astype('int'))


    train_dataset = MyDataset(X_train_resampled, y_train_resampled)
    test_dataset = MyDataset(X_test,y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader


train_data_loader, test_data_loader = create_data_loaders(new_df, 16)

# Save the predicted labels to a CSV file
import csv
output_file = 'predicted_labels.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sample', 'Predicted Label'])
    for i, label in enumerate(predicted_labels, start=1):
        writer.writerow([i, label])

print(f"Predicted labels saved to: {output_file}")

# Define hyperparameters
input_dim = 667
hidden_dim = 256
output_dim = 1
learning_rate = 0.0003
num_epochs = 10

# Create the LSTM model
model = ModifiedLSTMModel(input_dim, hidden_dim, output_dim)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare your training data (train_data_loader)

# Train the model
model.train()
for epoch in range(num_epochs):
    for inputs, targets in tqdm(train_data_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

import torch
import torch.nn as nn
import numpy as np

def evaluate(model, data_loader, device):
    model.eval()
    total_samples = 0
    correct_predictions = 0
    predicted_labels_list = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            predicted_labels = (torch.sigmoid(outputs) > 0.5).squeeze().long()

            total_samples += targets.size(0)
            correct_predictions += (predicted_labels == targets).sum().item()

            predicted_labels_list.append(predicted_labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    predicted_labels_array = np.concatenate(predicted_labels_list)
    return accuracy, predicted_labels_array

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate the model on the test data and get the predicted labels
accuracy, predicted_labels = evaluate(model, train_data_loader, device)
print(f"Accuracy: {accuracy:.4f}")

# Save the predicted labels to a file
output_file = 'summary_predicted_labels.npy'
np.save(output_file, predicted_labels)
print(f"Predicted labels saved to: {output_file}")