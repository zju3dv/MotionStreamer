import torch.nn as nn
from models.causal_cnn import CausalEncoder, CausalDecoder


# Causal TAE:
class Causal_TAE(nn.Module):
    def __init__(self,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 width=1024,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = []
                 ):
        
        super().__init__()

        self.decode_proj = nn.Linear(latent_dim, width)  

        self.encoder = CausalEncoder(272, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, latent_dim=latent_dim, clip_range=clip_range)
        self.decoder = CausalDecoder(272, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
    


    def preprocess(self, x):
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder, mu, logvar = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  
        
        return x_encoder, mu, logvar


    def forward(self, x):
        x_in = self.preprocess(x)       
        # Encode
        x_encoder, mu, logvar = self.encoder(x_in)  
        x_encoder = self.decode_proj(x_encoder) 
        # decoder
        x_decoder = self.decoder(x_encoder)
        x_out = self.postprocess(x_decoder)  
        return x_out, mu, logvar


    def forward_decoder(self, x):         
        # decoder
        x_width = self.decode_proj(x)           
        x_decoder = self.decoder(x_width)
        x_out = self.postprocess(x_decoder)
        return x_out


class Causal_HumanTAE(nn.Module):
    def __init__(self,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = []
                 ):
        
        super().__init__()
        self.tae = Causal_TAE(hidden_size, down_t, stride_t, hidden_size, depth, dilation_growth_rate, activation=activation, norm=norm, latent_dim=latent_dim, clip_range=clip_range)

    def encode(self, x):
        h, mu, logvar = self.tae.encode(x) 
        return h, mu, logvar

    def forward(self, x):
        x_out, mu, logvar = self.tae(x)
        return x_out, mu, logvar

    def forward_decoder(self, x):
        x_out = self.tae.forward_decoder(x)
        return x_out
        