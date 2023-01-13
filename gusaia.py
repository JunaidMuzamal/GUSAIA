import torch.nn as nn
from utils import *

device = get_default_device()

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, int(in_size / 2))
        self.fc2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.fc3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.fc1(w)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        mean = self.relu(out)
        log_var = self.relu(out)
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std) * std + mean
        return z, mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, int(out_size / 4))
        self.fc2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.fc3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        w = self.sigmoid(out)
        return w

class VAE(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder = Decoder(z_size, w_size)

    def forward(self, w):
        z, mean, log_var = self.encoder(w)
        w_hat = self.decoder(z)
        return w_hat, mean, log_var

    def training_step(self, batch):
        w_hat, mean, log_var = self(batch)
        loss = self.loss(batch, w_hat, mean, log_var)
        return loss

    def validation_step(self, batch):
        w_hat, mean, log_var = self(batch)
        loss = self.loss(batch, w_hat, mean, log_var)
        return {'val_loss': loss}

    def loss(self, w, w_hat, mean, log_var):
        reconstruction_loss = F.binary_cross_entropy(w_hat, w, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reconstruction_loss + kl_divergence

    def printLoss(self, epoch, result):
        print("Epoch {}, loss: {:.4f}".format(epoch, result['val_loss']))

    def fit(self, epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
        history = []
        optimizer = opt_func(self.parameters())
        for epoch in range(epochs):
            for batch in train_loader:
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = self.evaluate(val_loader)
            self.printLoss(epoch, result)
            history.append(result)
        return history
    def evaluate(self, val_loader):
        outputs = [self.validation_step(to_device(batch, device)) for batch in val_loader]
        return torch.stack([x['val_loss'] for x in outputs]).mean()
