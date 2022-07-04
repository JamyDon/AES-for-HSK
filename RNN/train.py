import torch
import torch.nn as nn
import torch.utils.data as data
from model import RNN
from config import config
import data_process
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from kappa import quadratic_weighted_kappa


def train(train_input, train_label, value_input, value_label):
    torch.manual_seed(0)

    train_dataset = data.TensorDataset(train_input, train_label)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    rnn = RNN(config)
    print(rnn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn = rnn.to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=config.lr)
    loss_func = nn.MSELoss()

    for epoch in range(config.epoch_num):
        for step, (batch_input, batch_label) in enumerate(train_loader):
            if batch_label.shape[0] < config.batch_size:
                continue

            batch_input, batch_label = batch_input.to(device), batch_label.to(device)
            out = rnn(batch_input)

            loss = loss_func(out, batch_label)
            y_pred = []
            out = torch.mul(out, config.essay_grade_num)
            out = torch.round(out)
            y_pred = out.long()
            # for _ in range(batch_label.shape[0]):
            #     pred_score = torch.argmax(out[_], dim=-1)
            #     # print(pred_score)
            #     y_pred.append(pred_score.long())
            batch_label = torch.mul(batch_label, config.essay_grade_num).long()

            qwk = quadratic_weighted_kappa(y_pred, batch_label, config.essay_grade_num)
            print('epoch ', epoch, ', step ', step,
                  ', loss = ', format(100*loss.item(), '.2f'),
                  ', qwk = ', format(qwk, '.2f'), sep='')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        value_input, value_label = value_input.to(device), value_label.to(device)
        value_out = rnn(value_input)
        value_out = torch.mul(value_out, config.essay_grade_num)
        value_out = torch.round(value_out)
        y_pred = value_out.long()
        # for _ in range(value_label.shape[0]):
        #     pred_score = config.essay_grade_num * torch.argmax(value_out[_], dim=-1)
        #     y_pred.append(pred_score.long())

        qwk = quadratic_weighted_kappa(y_pred, value_label, config.essay_grade_num)
        print('epoch ', epoch,
              ', value_qwk = ', format(qwk, '.2f'), sep='')
        print('--------------------------------------------')

    torch.save(rnn, 'rnn.pkl')


def test(test_input, test_label):
    rnn = torch.load('rnn.pkl')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn = rnn.to(device)

    test_input, test_label = test_input.to(device), test_label.to(device)
    test_out = rnn(test_input)
    test_out = torch.mul(test_out, config.essay_grade_num)
    test_out = torch.round(test_out)
    y_pred = test_out.long()
    # for _ in range(test_label.shape[0]):
    #     pred_score = torch.argmax(test_out[_], dim=-1) * config.essay_grade_num
    #     y_pred.append(pred_score)

    qwk = quadratic_weighted_kappa(y_pred, test_label, config.essay_grade_num)
    print('test_qwk = ', format(qwk, '.2f'), sep='')
    print('--------------------------------------------')


def main():
    train_input, train_label, value_input, value_label, test_input, test_label = data_process.load_data()

    train_label = torch.mul(train_label, 1 / config.essay_grade_num)

    train(train_input=train_input,
          train_label=train_label,
          value_input=value_input,
          value_label=value_label)

    test(test_input=test_input, test_label=test_label)


main()
