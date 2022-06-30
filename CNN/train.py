import torch
import torch.nn as nn
import torch.utils.data as data
from model import CNN
from config import Config
import data_process


def train(input_data, label):
    config = Config()
    torch.manual_seed(0)  # reproducible

    torch_dataset = data.TensorDataset(input_data, label)
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    cnn = CNN(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.lr)
    loss_func = nn.MSELoss()

    for epoch in range(config.epoch_num):
        for step, (batch_input, batch_label) in enumerate(loader):
            out = cnn(batch_input)
            print(out)
            print(batch_label)
            loss = loss_func(out, batch_label)
            print('step ', step, ', epoch ', epoch, 'loss = ', loss,sep="")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(cnn.state_dict(), 'cnn.pt')


input_data, label = data_process.divide_sentence('./data/split.csv')
train(input_data=input_data, label=label)
