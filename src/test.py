
#---------------------------------------------------------
# Information
#---------------------------------------------------------
__author__      = "biul"
__credits__    = ["none", "some"]
__maintainer__ = "biul"
__email__      = "qtrr9870@gmail.com"
__status__     = "Development"
__date__       = "2023.07.24"
__description__= "MLP model test for WiFi CSI data"

#---------------------------------------------------------
import argparse

import torch
import numpy as np
import time

# To use CSIKit, You should install csikit
# pip install csikit
# I will add the command in 'makefile'
from CSIKit.reader import get_reader
from CSIKit.util import csitools

#---------------------------------------------------------
def load_model( args ):
    device = torch.device('cpu')
    model = torch.load('./models/%s' %args.model, map_location=device)
    print('\n**************Load Mdoel**************\n')
    print(model)
    print('*****************************************')
    return model

#---------------------------------------------------------
def load_sample( args ):
    my_reader = get_reader('./samples/%s' %args.sample)
    csi_matrix = my_reader.read_file('./samples/%s' %args.sample, scaled=True)
    csi_data, no_frames, no_subcarriers = csitools.get_CSI(csi_matrix, metric='amplitude')

    csi_matrix_first = csi_data[:,:,0,0]
    csi_matrix_squeezed = np.squeeze(csi_matrix_first)
    x_test = np.delete(csi_matrix_squeezed, (0,1,2,3,4,5,11,25,28,29,30,31,32,33,34,35,36,38,52,58,59,60,61,62,63))
    x_test = np.where(x_test < -200, 0, x_test)
    print('Preprocessing for Outlier is Complete!\n')
    return x_test

#---------------------------------------------------------
def inferencing( x_test ):
    with torch.no_grad():
        device = torch.device('cpu')
        x_test = torch.tensor(x_test).float().to(device)
        
        output = model(x_test)

        labels = ['Blank', 'Left P', 'Left R', 'Mid P', 'Mid R', 'Right P', 'Right R']
        #prediction = labes[torch.argmax(output).item()]
        prediction = torch.argmax(output).item()
        print('Prediction : ', labels[prediction])
        
        #return will need for Application(I will use this result for keyboard interupt)
        #return prediction

#---------------------------------------------------------

if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser(description='HAR by WiFi CSI data')
        parser.add_argument('-m', '--model', type=str, default='None',
                            metavar='model', help='(str)name of the model that you want to use(default: None)')
        parser.add_argument('-s', '--sample', type=str, default='test_sample',
                            metavar='sample', help='(str)name of the sample that you want to test(default: test_sample)')
        args = parser.parse_args()
        return args

if __name__ == '__main__':
    args = get_args()

    model = load_model( args )

    while True:
        start = time.time()
        x_test = load_sample( args )
        inferencing( x_test )
        end = time.time()
        while (end - start) < 1 :
            end = time.time()
        print('*****************************************')
        print(f'All process is done in {end - start:.5f} sec!')
        print('*****************************************')
