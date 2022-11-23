import torch.nn as nn
from utils import *

device = get_default_device()


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, int(in_size / 2))
        self.layer2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.layer3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.layer1(w)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.layer1 = nn.Linear(latent_size, int(out_size / 4))
        self.layer2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.layer3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.layer1(z)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        w = self.sigmoid(out)
        return w


class GusaiaC(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        lossD1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        lossD2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return lossD1, lossD2

    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        lossD1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        lossD2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return {'D1_val_loss': lossD1, 'D2_val_loss': lossD2}

    def loss(self, outputs):
        batch_losses1 = [x['D1_val_loss'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['D2_val_loss'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'D1_val_loss': epoch_loss1.item(), 'D2_val_loss': epoch_loss2.item()}

    def printLoss(self, epoch, result):
        print("Epoch {}, D1_loss: {:.4f}, D2_loss: {:.4f}".format(epoch, result['D1_val_loss'], result['D2_val_loss']))


def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
    return model.loss(outputs)


def fit(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)

            # Training  AE1 in at this stage
            D1loss, D2loss = model.training_step(batch, epoch + 1)
            D1loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Training AE2 in at this stage
            D1loss, D2loss = model.training_step(batch, epoch + 1)
            D2loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch + 1)
        model.printLoss(epoch, result)
        history.append(result)
    return history


def validate(model, test_loader, alpha=.5, beta=.5):
    results = []
    for [batch] in test_loader:
        batch = to_device(batch, device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        results.append(alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1))
    return results
