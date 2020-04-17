import torch
import torch.nn as nn
import torch.nn.functional as F

# # Device configuration
device = torch.device('cuda', 0)

class Q_predictor(nn.Module):
    def __init__(self, hidden_size, output_size, input_size, num_layers, bidirectional = True):
        super(Q_predictor, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                          dropout=0, bidirectional=bidirectional, batch_first = True)
        self.bi_directional = bidirectional
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim = 2)


    def forward(self, input_,hidden):
        h0 = hidden[0]#torch.zeros(self.num_layers, input_.size(0), self.hidden_size).to(device)
        c0 = hidden[1]#torch.zeros(self.num_layers, input_.size(0), self.hidden_size).to(device)
        #output = F.relu(input_)
        if self.bi_directional:
            out_, hid_ = self.lstm(F.relu(input_), (h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
            output = out_[:,:,self.hidden_size:] + out_[:,:,:self.hidden_size]
            # Decode the hidden state of the last time step
            output = self.softmax(self.out(output))
            return output, hid_
        #return self.softmax(self.selector_Q_2(self.Dp(self.selector_Q_1(input_))))

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, drop_out = 0, bidirectional = False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, self.num_layers,
                          dropout=drop_out, bidirectional=bidirectional, batch_first = True)
        self.bi_directional = bidirectional
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_, hidden):
        h0 = hidden[0]#torch.zeros(self.num_layers, input_.size(0), self.hidden_size).to(device)
        c0 = hidden[1]#torch.zeros(self.num_layers, input_.size(0), self.hidden_size).to(device)
        output = self.embedding(input_).view(1,1,-1)
        if self.bi_directional:
            out_, hid_ = self.lstm(output, (h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
            output = out_[:,:,self.hidden_size:] + out_[:,:,:self.hidden_size]
            # Decode the hidden state of the last time step
            output = self.softmax(self.out(output))
            return output, hid_
        else:
            output, hidden = self.lstm(output, (h0,c0))
            output = self.softmax(self.out(output))
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, maxlength, drop_out = 0, bidirectional = False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_length = maxlength
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, self.num_layers,
                          dropout=drop_out, bidirectional=bidirectional, batch_first = True)
        self.bi_directional = bidirectional
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_, hidden, encoderhid):
        h0 = hidden[0]#torch.zeros(self.num_layers, input_.size(0), self.hidden_size).to(device)
        c0 = hidden[1]#torch.zeros(self.num_layers, input_.size(0), self.hidden_size).to(device)
        embedded = self.embedding(input_).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output, hidden = self.lstm(output, (h0,c0))
        output = F.log_softmax(self.out(output))
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Recurrent neural network (many-to-one)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bi_directional = False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          dropout=0, bidirectional=bi_directional, batch_first = True)#nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bi_directional = bi_directional
        if self.bi_directional:
            self.multiplier = 2
        else:
            self.multiplier = 1

    def forward(self, x, hidden):
        # Set initial hidden and cell states
        h0 = hidden[0]#torch.zeros(self.multiplier*self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = hidden[1]#torch.zeros(self.multiplier*self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        if self.bi_directional:
            out_, hid_ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
            out = out_[:,:,self.hidden_size:] + out_[:,:,:self.hidden_size]
            # Decode the hidden state of the last time step
            return hid_
        else:
            out, hid_ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
            # Decode the hidden state of the last time step
            return hid_, out
