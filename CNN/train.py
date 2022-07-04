import torch
import torch.nn as nn
import torch.utils.data as data
from model import CNN
from config import Config
import data_process
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from kappa import quadratic_weighted_kappa


def train(input_data, label):
    config = Config()
    torch.manual_seed(0)

    torch_dataset = data.TensorDataset(input_data, label)
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    cnn = CNN(config)
    print(cnn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(config.epoch_num):
        for step, (batch_input, batch_label) in enumerate(loader):
            if batch_label.shape[0] < config.batch_size:
                continue

            batch_input, batch_label = batch_input.to(device), batch_label.to(device)
            out = cnn(batch_input)
            # print(out)
            # print(batch_label)
            loss = loss_func(out, batch_label)
            correct_cnt = 0
            y_pred = []
            for _ in range(batch_label.shape[0]):
                y_pred.append(torch.argmax(out[_], dim=-1))
                if torch.argmax(out[_], dim=-1) == batch_label[_]:
                    correct_cnt += 1
            acc = correct_cnt / batch_label.shape[0]
            qwk = quadratic_weighted_kappa(y_pred, batch_label, config.essay_grade_num)
            if step == 88:
                print('epoch ', epoch, ', test_loss = ', format(loss.item(), '.2f'), ', test_acc = ', acc,
                      ', test_qwk = ', qwk, sep='')
                continue
            print('epoch ', epoch, ', step ', step, ', loss = ', format(loss.item(), '.2f'), ', qwk = ', qwk, sep="")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(cnn.state_dict(), 'cnn.pt')


input_data, label = data_process.divide_sentence('../data/split.csv')
train(input_data=input_data, label=label)
