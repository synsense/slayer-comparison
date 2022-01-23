import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/..")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from learningStats import learningStats
import zipfile
from slayer_layer import SlayerLayer

netParams = snn.params("network.yaml")


def augmentData(event):
    xs = 8
    ys = 8
    xjitter = np.random.randint(2 * xs) - xs
    yjitter = np.random.randint(2 * ys) - ys
    event.x += xjitter
    event.y += yjitter
    return event


# Dataset definition
class nmnistDataset(Dataset):
    def __init__(
        self, datasetPath, sampleFile, samplingTime, sampleLength, augment=False
    ):
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype("int")
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)
        self.augment = augment

    def __getitem__(self, index):
        inputIndex = self.samples[index, 0]
        classLabel = self.samples[index, 1]

        event = snn.io.read2Dspikes(self.path + str(inputIndex.item()) + ".bs2")
        if self.augment is True:
            event = augmentData(event)
        inputSpikes = event.toSpikeTensor(
            torch.zeros((2, 34, 34, self.nTimeBins)), samplingTime=self.samplingTime
        )

        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel, ...] = 1
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]


# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = SlayerLayer(netParams["neuron"], netParams["simulation"])
        self.slayer = slayer

        # weight normalization
        self.conv1 = torch.nn.utils.weight_norm(
            slayer.conv(2, 16, 5, padding=1), name="weight"
        )
        self.conv2 = torch.nn.utils.weight_norm(
            slayer.conv(16, 32, 3, padding=1), name="weight"
        )
        self.conv3 = torch.nn.utils.weight_norm(
            slayer.conv(32, 64, 3, padding=1), name="weight"
        )

        self.pool1 = slayer.pool(2)
        self.pool2 = slayer.pool(2)

        self.fc1 = torch.nn.utils.weight_norm(
            slayer.dense((8 * 8 * 64), 512), name="weight"
        )
        self.fc2 = torch.nn.utils.weight_norm(slayer.dense(512, 10), name="weight")

        # delays
        self.delay1 = slayer.delay(16)
        self.delay2 = slayer.delay(16)
        self.delay3 = slayer.delay(32)
        self.delay4 = slayer.delay(32)
        self.delay5 = slayer.delay(64 * 8 * 8)
        self.delay6 = slayer.delay(512)

    def forward(self, spike):
        # count.append(torch.sum(spike).item())

        spike = self.slayer.spike(self.conv1(self.slayer.psp(spike)))  # 32, 32, 16
        spike = self.delay1(spike)

        spike = self.slayer.spike(self.pool1(self.slayer.psp(spike)))  # 16, 16, 16
        spike = self.delay2(spike)

        spike = self.slayer.spike(self.conv2(self.slayer.psp(spike)))  # 16, 16, 32
        spike = self.delay3(spike)

        spike = self.slayer.spike(self.pool2(self.slayer.psp(spike)))  # 8,  8, 32
        spike = self.delay4(spike)

        spike = self.slayer.spike(self.conv3(self.slayer.psp(spike)))  # 8,  8, 64
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))
        spike = self.delay5(spike)

        spike = self.slayer.spike(self.fc1(self.slayer.psp(spike)))  # 10
        spike = self.delay6(spike)

        spike = self.slayer.spike(self.fc2(self.slayer.psp(spike)))  # 10

        return spike

    def clamp(self):
        self.delay1.delay.data.clamp_(0, 64)
        self.delay2.delay.data.clamp_(0, 64)
        self.delay3.delay.data.clamp_(0, 64)
        self.delay4.delay.data.clamp_(0, 64)
        self.delay5.delay.data.clamp_(0, 64)
        self.delay6.delay.data.clamp_(0, 64)

    def gradFlow(self, path):
        gradNorm = lambda x: torch.norm(x).item() / torch.numel(x)

        grad = []
        grad.append(gradNorm(self.conv1.weight_g.grad))
        grad.append(gradNorm(self.conv2.weight_g.grad))
        grad.append(gradNorm(self.conv3.weight_g.grad))
        grad.append(gradNorm(self.fc1.weight_g.grad))
        grad.append(gradNorm(self.fc2.weight_g.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + "gradFlow.png")
        plt.close()


if __name__ == "__main__":
    # # Extract NMNIST samples
    # with zipfile.ZipFile("NMNISTsmall.zip") as zip_file:
    #     for member in zip_file.namelist():
    #         if not os.path.exists("./" + member):
    #             zip_file.extract(member, "./")

    device = torch.device("cuda")
    net = Network(netParams).to(device)
    error = snn.loss(netParams).to(device)

    # Custom NADAM optimizer
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr=0.01, amsgrad=False)

    # Dataset and dataLoader instances.
    trainingSet = nmnistDataset(
        datasetPath=netParams["training"]["path"]["in"],
        sampleFile=netParams["training"]["path"]["train"],
        samplingTime=netParams["simulation"]["Ts"],
        sampleLength=netParams["simulation"]["tSample"],
    )
    trainLoader = DataLoader(
        dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4
    )

    testingSet = nmnistDataset(
        datasetPath=netParams["training"]["path"]["in"],
        sampleFile=netParams["training"]["path"]["test"],
        samplingTime=netParams["simulation"]["Ts"],
        sampleLength=netParams["simulation"]["tSample"],
    )
    testLoader = DataLoader(
        dataset=testingSet, batch_size=12, shuffle=False, num_workers=4
    )

    # Learning stats instance.
    stats = learningStats()

    # # Visualize the network.
    # for i in range(5):
    #   input, target, label = trainingSet[i]
    #   snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))

    # training loop
    for epoch in range(200):
        tSt = datetime.now()

        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
            # Move the input and target to correct GPU.
            input = input.to(device)
            target = target.to(device)

            # Forward pass of the network.
            output = net.forward(input)

            # Gather the training stats.
            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            stats.training.numSamples += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            # Reset gradients to zero.
            optimizer.zero_grad()

            # Backward pass of the network.
            loss.backward()

            # Update weights.
            optimizer.step()

            # Clamp delay
            net.clamp()

            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        # Testing loop.
        # Same steps as Training loops except loss backpropagation and weight update.
        for i, (input, target, label) in enumerate(testLoader, 0):
            input = input.to(device)
            target = target.to(device)

            output = net.forward(input)

            stats.testing.correctSamples += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            stats.testing.numSamples += len(label)

            loss = error.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            stats.print(epoch, i)

        # Update stats.
        stats.update()

    # Plot the results.
    plt.figure(1)
    plt.semilogy(stats.training.lossLog, label="Training")
    plt.semilogy(stats.testing.lossLog, label="Testing")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.figure(2)
    plt.plot(stats.training.accuracyLog, label="Training")
    plt.plot(stats.testing.accuracyLog, label="Testing")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
