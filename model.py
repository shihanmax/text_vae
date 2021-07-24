import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Highway(nn.Module):
    
    def __init__(self, n_highway_layers, embedding_dim):
        super(Highway, self).__init__()
        self.n_layers = n_highway_layers 
        
        self.non_linear = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) 
             for _ in range(self.n_layers)]
        )
        self.linear = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) 
             for _ in range(self.n_layers)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) 
             for _ in range(self.n_layers)]
        )
        
    def forward(self, x):
        for i in range(self.n_layers):
            gate = torch.sigmoid(self.gate[i](x))
            non_linear_out = torch.relu(self.non_linear[i](x))
            linear_out = self.linear[i](x)
            
            x = gate * non_linear_out + (1 - gate) * linear_out
        return x
    

class Encoder(nn.Module):
    
    def __init__(
        self, n_highway_layers, embedding_dim, encoder_hidden_dim, 
        encoder_num_layers, z_dim,
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=100,
            embedding_dim=embedding_dim,
        )
        
        self.highway = Highway(n_highway_layers, embedding_dim)
        self.num_layers = encoder_num_layers
        self.hidden_size = encoder_hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=encoder_hidden_dim, 
            num_layers=encoder_num_layers,
            bidirectional=True,
            batch_first=True,
        )
                
        self.z2mean = nn.Linear(2 * encoder_hidden_dim, z_dim)
        self.z2log_var = nn.Linear(2 * encoder_hidden_dim, z_dim)
        self.fc_out = nn.Linear(2 * encoder_hidden_dim, 2 * encoder_hidden_dim)

    def sample(self, hn):
        noise = torch.rand_like(hn)
        
        # re-parameterize
        mean = self.z2mean(hn)
        log_var = self.z2log_var(hn)
        
        z = mean + noise * torch.exp(0.5 * log_var)

        return mean, log_var, z

    def forward(self, x, valid_length):
        bs, *_ = x.shape
        x = self.highway(x)
        
        packed_sequence = pack_padded_sequence(
            x, length=valid_length, batch_first=True, enforce_sorted=False,
        )

        _, (hn, _) = self.lstm(packed_sequence)
        
        hn = hn.view(self.num_layers, 2, bs, self.hidden_size)
        
        # get hn of last (top) rnn layer
        hn = hn[-1]  # num_direc, bs, hidden_dim
        
        # concat hidden of both directions
        hn = hn.permute(1, 0, 2).contiguous().view(bs, -1)  # bs, hidden * 2
        
        hn = nn.ReLU(self.fc_out(hn))

        z, mean, log_var = self.sample(hn)
        
        return z, mean, log_var
        

class Generator(nn.Module):
    
    def __init__(
        self, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, 
        decoder_num_layers, vocab_size, z_dim, max_decode_len,
    ):

        super(Generator, self).__init__()

        self.max_decode_len = max_decode_len
        
        self.embedding = nn.Embedding(
            num_embeddings=100,
            embedding_dim=embedding_dim,
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim + z_dim, 
            hidden_size=decoder_hidden_dim, 
            num_layers=decoder_num_layers,
            batch_first=True,
        )

        self.multi_fc = nn.Sequential(
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(decoder_hidden_dim, vocab_size),
        )

        self.fc_after_rnn = nn.Linear(decoder_hidden_dim + z_dim, vocab_size)

    def forward(self, latent_z):

        input_ = None  # <SOS>
        hidden = None
        
        decoding_result = []
        
        outputs = []
        
        for _ in range(self.max_decode_len):
            out, hidden = self.step(hidden, latent_z, input_)
            outputs.append(out)
            
            max_val, indices = torch.max(out)
            decoding_result.append(indices)
        
        indices = torch.cat(decoding_result, dim=0)
        outputs = torch.cat(outputs, dim=-1)  # differentiable
        
        print("generate indices:{}".format(indices.shape))
        print("output: {}".format(outputs.shape))
        
        return indices
        
    def step(self, hidden, latent_z, input_):
        input_ = self.embedding(input_)
        
        out, hidden = self.lstm(torch.cat([latent_z, input_]), hidden)
        out = torch.cat([latent_z, out])  # cat out and latent_z
        out = self.fc_after_rnn(out)  # linear transform
        
        return out, hidden
    

class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()
    
    def forward(self):
        pass
    
