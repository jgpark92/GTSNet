import torch
import torch.nn as nn
import torch.nn.functional as F

class HAR_ConvLSTM(nn.Module):
    def __init__(self, nc_input, n_classes):
        super(HAR_ConvLSTM, self).__init__()

        self.conv1 = nn.Conv1d(nc_input, 64, 5)
        self.conv2 = nn.Conv1d(64, 64, 5)
        self.conv3 = nn.Conv1d(64, 64, 5)
        self.conv4 = nn.Conv1d(64, 64, 5)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        self.fc = nn.Linear(128, n_classes)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.lstm(x)

        h = h[-1,:,:]
        h = self.dropout(h)
        logits = self.fc(h)

        return logits


class HAR_BiLSTM(nn.Module):
    def __init__(self, nc_input, n_classes):
        super(HAR_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=nc_input, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(128*2, n_classes)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.lstm(x)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        logits = self.fc(h)

        return logits

class HAR_LSTM(nn.Module):
    def __init__(self, nc_input, n_classes):
        super(HAR_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=nc_input, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.lstm(x)
        h = h[-1,:,:]
        logits = self.fc(h)

        return logits
