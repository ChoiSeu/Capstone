import argparse

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from CSIKit.reader import get_reader
from CSIKit.util import csitools

def dataloader( args ):
    name1 = f'./samples/train_data/{args.date}/{args.date}_blank%d.pcap'
    name2 = f'./samples/train_data/{args.date}/{args.date}_leftpa%d.pcap'
    name3 = f'./samples/train_data/{args.date}/{args.date}_leftro%d.pcap'
    name4 = f'./samples/train_data/{args.date}/{args.date}_middlepa%d.pcap'
    name5 = f'./samples/train_data/{args.date}/{args.date}_middlero%d.pcap'
    name6 = f'./samples/train_data/{args.date}/{args.date}_rightpa%d.pcap'
    name7 = f'./samples/train_data/{args.date}/{args.date}_rightro%d.pcap'

    my_reader1 = get_reader(name1 %1)
    csi_matrix1 = my_reader1.read_file(name1 %1, scaled=True)
    csi_data1, no_frames1, no_subcarriers1 = csitools.get_CSI(csi_matrix1, metric="amplitude")

    my_reader2 = get_reader(name2 %1)
    csi_matrix2 = my_reader2.read_file(name2 %1, scaled=True)
    csi_data2, no_frames2, no_subcarriers2 = csitools.get_CSI(csi_matrix2, metric="amplitude")

    my_reader3 = get_reader(name3 %1)
    csi_matrix3 = my_reader3.read_file(name3 %1, scaled=True)
    csi_data3, no_frames3, no_subcarriers3 = csitools.get_CSI(csi_matrix3, metric="amplitude")

    my_reader4 = get_reader(name4 %1)
    csi_matrix4 = my_reader4.read_file(name4 %1, scaled=True)
    csi_data4, no_frames4, no_subcarriers4 = csitools.get_CSI(csi_matrix4, metric="amplitude")

    my_reader5 = get_reader(name5 %1)
    csi_matrix5 = my_reader5.read_file(name5 %1, scaled=True)
    csi_data5, no_frames5, no_subcarriers5 = csitools.get_CSI(csi_matrix5, metric="amplitude")

    my_reader6 = get_reader(name6 %1)
    csi_matrix6 = my_reader6.read_file(name6 %1, scaled=True)
    csi_data6, no_frames6, no_subcarriers6 = csitools.get_CSI(csi_matrix6, metric="amplitude")

    my_reader7 = get_reader(name7 %1)
    csi_matrix7 = my_reader7.read_file(name7 %1, scaled=True)
    csi_data7, no_frames7, no_subcarriers7 = csitools.get_CSI(csi_matrix7, metric="amplitude")

    for i in range (2, args.num):
        my_reader = get_reader(name1 %i)
        temp = my_reader.read_file(name1 %i, scaled=True)
        tmp0, tmp1, tmp2 = csitools.get_CSI(temp, metric="amplitude")
        csi_data1 = np.concatenate((csi_data1, tmp0), axis = 0)

    for i in range (2, args.num):
        my_reader = get_reader(name2 %i)
        temp = my_reader.read_file(name2 %i, scaled=True)
        tmp0, tmp1, tmp2 = csitools.get_CSI(temp, metric="amplitude")
        csi_data2 = np.concatenate((csi_data2, tmp0), axis = 0)

    for i in range (2, args.num):
        my_reader = get_reader(name3 %i)
        temp = my_reader.read_file(name3 %i, scaled=True)
        tmp0, tmp1, tmp2 = csitools.get_CSI(temp, metric="amplitude")
        csi_data3 = np.concatenate((csi_data3, tmp0), axis = 0)

    for i in range (2, args.num):
        my_reader = get_reader(name4 %i)
        temp = my_reader.read_file(name4 %i, scaled=True)
        tmp0, tmp1, tmp2 = csitools.get_CSI(temp, metric="amplitude")
        csi_data4 = np.concatenate((csi_data4, tmp0), axis = 0)

    for i in range (2, args.num):
        my_reader = get_reader(name5 %i)
        temp = my_reader.read_file(name5 %i, scaled=True)
        tmp0, tmp1, tmp2 = csitools.get_CSI(temp, metric="amplitude")
        csi_data5 = np.concatenate((csi_data5, tmp0), axis = 0)

    for i in range (2, args.num):
        my_reader = get_reader(name6 %i)
        temp = my_reader.read_file(name6 %i, scaled=True)
        tmp0, tmp1, tmp2 = csitools.get_CSI(temp, metric="amplitude")
        csi_data6 = np.concatenate((csi_data6, tmp0), axis = 0)

    for i in range (2, args.num):
        my_reader = get_reader(name7 %i)
        temp = my_reader.read_file(name7 %i, scaled=True)
        tmp0, tmp1, tmp2 = csitools.get_CSI(temp, metric="amplitude")
        csi_data7 = np.concatenate((csi_data7, tmp0), axis = 0)

    csi_label1 = np.zeros(len(csi_data1),dtype = np.int32)
    csi_label2 = np.ones(len(csi_data2),dtype = np.int32)
    csi_label3 = np.full(len(csi_data3),2,dtype = np.int32)
    csi_label4 = np.full(len(csi_data4),3,dtype = np.int32)
    csi_label5 = np.full(len(csi_data5),4,dtype = np.int32)
    csi_label6 = np.full(len(csi_data6),5,dtype = np.int32)
    csi_label7 = np.full(len(csi_data7),6,dtype = np.int32)

    datas = np.concatenate((csi_data1, csi_data2, csi_data3, csi_data4, csi_data5, csi_data6, csi_data7), axis = 0)
    labels = np.concatenate((csi_label1, csi_label2, csi_label3, csi_label4, csi_label5, csi_label6, csi_label7), axis = 0)

    csi_matrix_first = datas[:, :, 0, 0]
    csi_matrix_squeezed = np.squeeze(csi_matrix_first)
    csi_matrix_squeezed = np.delete(csi_matrix_squeezed, (0,1,2,3,4,5,11,25,28,29,30,31,32,33,34,35,36,38,52,58,59,60,61,62,63), 1)
    
    data_length = csi_matrix_squeezed.shape[0]
    data_id     = np.arange(0, data_length)
    np.random.shuffle(data_id)  # 데이터인덱스 셔플링

    train_size = 0.8
    train_id   = data_id[0:int(train_size*data_length)]  # 앞부분에 해당하는 인덱스셋 가져오기(학습)
    test_id    = data_id[int(train_size*data_length)::]  # 뒷부분에 해당하는 인덱스셋 가져오기(테스트)

    x_train     = csi_matrix_squeezed[train_id] # 학습 인덱스셋에 해당하는 데이터셋 가져오기
    y_train     = labels[train_id] # 학습 인덱스셋에 해당하는 레이블셋 가져오기
    x_test      = csi_matrix_squeezed[test_id]  # 테스트 인덱스셋에 해당하는 데이터셋 가져오기
    y_test      = labels[test_id]  # 테스트 인덱스셋에 해당하는 레이블셋 가져오기

    x_train = np.where(x_train < -200, 0, x_train)
    x_test = np.where(x_test < -200, 0, x_test)

    delete_list1 = []
    for i in range(0,len(x_train)):
        if x_train[i,0] == 0:
            delete_list1.append(i)

    delete_list2 = []
    for i in range(0,len(x_test)):
        if x_test[i,0] == 0:
            delete_list2.append(i)

    x_train = np.delete(x_train, delete_list1, axis = 0)
    y_train = np.delete(y_train, delete_list1,)
    x_test = np.delete(x_test, delete_list2, axis = 0)
    y_test = np.delete(y_test, delete_list2,)

    x_train = np.floor(x_train.reshape(-1, 39) * (-1))
    x_test = np.floor(x_test.reshape(-1, 39) * (-1))
    
    batch_size = args.batch_size
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).type(torch.LongTensor))
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).type(torch.LongTensor))
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

    return train_data_loader, x_test, y_test

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser(description='HAR by WiFi CSI data')
        parser.add_argument('-d', '--date', type=str, default='0725', metavar='date', help='(str)date of the samples that you want to use(default: 0725)')
        parser.add_argument('-n', '--num', type=int, default=10, metavar='num', help='(int)per number of the samples that you want to use(default: 10)')
        parser.add_argument('-b', '--batch_size', type=int, default=20, metavar='batch_size', help='(int)batch size of the dataloader(default: 20)')
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='epochs')
        parser.add_argument('-l', '--lr', type=float, default=0.001, metavar='lr')
        parser.add_argument('-m', '--more', type=str2bool, default=False, metavar='more', help='(bool)If you want to train more, set this arg True(default: False)')
        parser.add_argument('-g', '--goal', type=float, default=0.9, metavar='goal')
        parser.add_argument('-p', '--pretrained_model', type=str, default='None', metavar='pretrained_model', help='(str)If you want to improve or finetuning with pretrained model, set this arg(default: None)')
        args = parser.parse_args()
        return args

if __name__ == '__main__':
    args = get_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    
    train_data_loader, x_test, y_test = dataloader( args )
    if args.more == False:
        FC1 = torch.nn.Linear(39, 128, bias=True)
        FC2 = torch.nn.Linear(128, 128, bias=True)
        FC3 = torch.nn.Linear(128, 64, bias=True)
        FC4 = torch.nn.Linear(64, 7, bias=True)
        elu = torch.nn.ELU()
        model = torch.nn.Sequential(FC1, elu, FC2, elu, FC3, elu, FC4).to(device)
    else:
        model = torch.load('./models/%s' %args.pretrained_model)

    # Initialize the loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)  # 내부에 Softmax를 포함함
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        avg_cost = 0
        total_batch = len(train_data_loader)
        
        for X, T in train_data_loader:
            X = X.to(device)
            T = T.to(device)
            optimizer.zero_grad()   # 기존 계산한 경사값 삭제
            output = model(X)       # 순방향 연산
            cost = criterion(output, T)
            cost.backward()         # 경사값 계산
            optimizer.step()        # 업데이트 1회 수행
            avg_cost += cost / total_batch     # 평균 손실함수값 계산

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        with torch.no_grad():
            x_test = torch.tensor(x_test).float().to(device)
            T_test = torch.tensor(y_test).to(device)
            # 테스트셋에 대해 추론 수행
            output = model(x_test)
            # 출력값이 가장 높은 뉴런의 인덱스와 정답을 비교, 맞으면 1, 틀리면 0
            correct_prediction = torch.argmax(output, 1) == T_test
            # 정확도 계산
            accuracy = correct_prediction.float().mean()
            print('Accuracy:', accuracy.item())
            print('----------------------------------------------\n')

        #일정 정확도 달성 후 종료
        if accuracy.item() > args.goal :
            break
    torch.save(model, f'./models/checkpoint_acc_{accuracy:.2f}.pth')
    print('Learning finished')

