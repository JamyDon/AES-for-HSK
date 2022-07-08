import torch
import torch.nn as nn
import torch.utils.data as data
from model import AttentionNN
from config import config
import data_process
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from kappa import quadratic_weighted_kappa


model_path_loss = 'attention_nn_loss.pkl'
model_path_qwk = 'attention_nn_qwk.pkl'


def train(train_input, train_label, value_input, value_label):
    torch.manual_seed(0)

    train_dataset = data.TensorDataset(train_input, train_label)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    network = AttentionNN(config)
    print(network)
    total_params = sum(p.numel() for p in network.parameters())
    print('total_params = ', total_params, sep='')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)
    loss_func = nn.MSELoss()

    max_value_qwk = 0
    min_value_loss = 1e5

    for epoch in range(config.epoch_num):
        step_cnt = 0
        train_loss = 0
        train_qwk = 0

        for step, (batch_input, batch_label) in enumerate(train_loader):
            if batch_label.shape[0] < config.batch_size:
                continue

            step_cnt += 1
            batch_input, batch_label = batch_input.to(device), batch_label.to(device)
            out = network(batch_input)

            loss = loss_func(out, batch_label)
            train_loss += loss.item()
            y_pred = []
            out = torch.mul(out, config.essay_grade_num)
            out = torch.floor(out)
            y_pred = out.long()
            # for _ in range(batch_label.shape[0]):
            #     pred_score = torch.argmax(out[_], dim=-1)
            #     # print(pred_score)
            #     y_pred.append(pred_score.long())
            batch_label = torch.mul(batch_label, config.essay_grade_num).long()

            qwk = quadratic_weighted_kappa(y_pred, batch_label, config.essay_grade_num)
            train_qwk += qwk
            # print('epoch ', epoch+1, ', step ', step+1,
            #       ', loss = ', format(100*loss.item(), '.3f'),
            #       ', qwk = ', format(qwk, '.3f'), sep='')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= step_cnt
        train_loss *= 100
        train_qwk /= step_cnt
        print('epoch ', epoch+1,
              ', train_loss = ', format(train_loss, '.3f'),
              ', train_qwk = ', format(train_qwk, '.3f'), sep='')

        with torch.no_grad():
            value_input, value_label = value_input.to(device), value_label.to(device)

            value_dataset = data.TensorDataset(value_input, value_label)
            value_loader = data.DataLoader(
                dataset=value_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
            )

            step_cnt = 0
            loss_sum = 0
            qwk_sum = 0

            for step, (batch_input, batch_label) in enumerate(value_loader):
                if batch_label.shape[0] < config.batch_size:
                    continue

                step_cnt += 1
                batch_input, batch_label = batch_input.to(device), batch_label.to(device)
                out = network(batch_input)
                loss_sum += loss_func(out, batch_label)
                out = torch.mul(out, config.essay_grade_num)
                out = torch.floor(out)
                y_pred = out.long()
                batch_label = torch.mul(batch_label, config.essay_grade_num).long()
                qwk = quadratic_weighted_kappa(y_pred, batch_label, config.essay_grade_num)
                qwk_sum += qwk

            value_loss = loss_sum / step_cnt * 100
            qwk = qwk_sum / step_cnt

            print('epoch ', epoch+1,
                  ', value_loss = ', format(value_loss, '.3f'),
                  ', value_qwk = ', format(qwk, '.3f'), sep='')
            print('--------------------------------------------')

            if value_loss < min_value_loss:
                min_value_loss = value_loss
                torch.save(network, model_path_loss)
            if qwk > max_value_qwk:
                max_value_qwk = qwk
                torch.save(network, model_path_qwk)


def test(test_input, test_label):
    model_paths = [model_path_loss, model_path_qwk]
    for model_path in model_paths:
        network = torch.load(model_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network = network.to(device)

        with torch.no_grad():
            test_input, test_label = test_input.to(device), test_label.to(device)

            test_dataset = data.TensorDataset(test_input, test_label)
            test_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
            )

            step_cnt = 0
            loss_sum = 0
            qwk_sum = 0

            for step, (batch_input, batch_label) in enumerate(test_loader):
                if batch_label.shape[0] < config.batch_size:
                    continue

                step_cnt += 1
                batch_input, batch_label = batch_input.to(device), batch_label.to(device)
                out = network(batch_input)
                out = torch.mul(out, config.essay_grade_num)
                out = torch.floor(out)
                y_pred = out.long()
                batch_label = torch.mul(batch_label, config.essay_grade_num).long()
                qwk = quadratic_weighted_kappa(y_pred, batch_label, config.essay_grade_num)
                qwk_sum += qwk

            qwk = qwk_sum / step_cnt

            print('test_qwk = ', format(qwk, '.3f'), sep='')
            print('--------------------------------------------')


def main():
    train_input, train_label, value_input, value_label, test_input, test_label = data_process.load_data()

    train_label = torch.mul(train_label, 1 / config.essay_grade_num)
    value_label = torch.mul(value_label, 1 / config.essay_grade_num)
    test_label = torch.mul(test_label, 1 / config.essay_grade_num)

    train(train_input=train_input,
          train_label=train_label,
          value_input=value_input,
          value_label=value_label)

    test(test_input=test_input, test_label=test_label)


main()
