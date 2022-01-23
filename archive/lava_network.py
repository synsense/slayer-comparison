import torch
import pytorch_lightning as pl
import lava.lib.dl.slayer as slayer


class LavaNetwork(pl.LightningModule):
    def __init__(self, voltage_decay=0.1, spike_threshold=0.1, learning_rate=1e-5, weight_decay=None):
        super(LavaNetwork, self).__init__()
        self.save_hyperparameters()

        neuron_params = {
                'threshold'     : spike_threshold,
                'current_decay' : 1,
                'voltage_decay' : voltage_decay,
                'requires_grad' : True,
                # 'scale'         : 1<<4,
            }
        
        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(neuron_params, 200, 200),
                # slayer.block.cuba.Dense(neuron_params, 256, 200),
            ])

        self.error = slayer.loss.SpikeTime(time_constant=2, filter_order=2)
    
    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()
        x, y = batch
        y_hat = self(x)
        loss = self.error(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

