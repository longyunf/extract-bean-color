import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
from os.path import join 
import numpy as np
np.random.seed(1)
import argparse

import matplotlib.pyplot as plt
from tensorflow import keras

import _init_paths
from data_generator import DataGenerator
from functions import loss, cbk_comp_ap
import netStructure


def mkdir(dir1):
    if not os.path.exists(dir1): 
        os.makedirs(dir1)
        print('make directory %s' % dir1)
        

def main(args):    
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data') + os.sep
        
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result')              
    mkdir(args.dir_result)     
    
    dir_result=args.dir_result
    
    dir_train=args.dir_data+'train_patch'+os.sep
    dir_val=args.dir_data+'val_patch'+os.sep
    
    dir_im_train=dir_train+'image_rot'+os.sep
    dir_label_train=dir_train+'label_rot'+os.sep
    dir_msk_train=dir_train+'msk_rot'+os.sep
    
    dir_im_val=dir_val+'image'+os.sep         
    dir_label_val=dir_val+'label'+os.sep      
    dir_msk_val=dir_val+'msk'+os.sep         
        
    epochs = 20    
    h=128
    w=128
        
    params_train = {'dir_input': dir_im_train,
                    'dir_label': dir_label_train, 
                    'dir_msk': dir_msk_train,      
                    'batch_size': 16,
                    'dim_input': (h,w,3),
                    'dim_label': (h,w,1),
                    'shuffle': True}
    
    params_val = {'dir_input': dir_im_val,
                    'dir_label': dir_label_val, 
                    'dir_msk': dir_msk_val,       
                    'batch_size': 16,
                    'dim_input': (h,w,3),
                    'dim_label': (h,w,1),
                    'shuffle': True}
    
    params_comp_AP = {'dir_im_train': dir_train+'image'+os.sep,              
                      'dir_im_val': dir_im_val,
                      'dir_label_train': dir_train+'label'+os.sep,
                      'dir_label_val': dir_label_val,
                      'dir_msk_train': dir_train+'msk'+os.sep,
                      'dir_msk_val': dir_msk_val,
                      'class_id': 1}
       
    train_generator = DataGenerator(**params_train)
    val_generator = DataGenerator(**params_val)
    
    
    print('input dimension:', train_generator.dim_input)
    print(len(train_generator.list_IDs), 'train samples')
    print(len(val_generator.list_IDs), 'test samples')
        
    input_shape=train_generator.dim_input
    
    model = netStructure.net(input_shape)
    
    model.summary()
        
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=None)
    
    cbk_ap = cbk_comp_ap(**params_comp_AP)
    
    history=model.fit_generator(generator=train_generator,
                        callbacks=[cbk_ap],
                        validation_data=val_generator,
                        epochs=epochs,
                        verbose=1,
                        use_multiprocessing=False,
                        workers=3)
    
    fig=plt.figure(1)
    plt.plot(history.history['loss'],'-*r')
    plt.plot(history.history['val_loss'],'-*b')
    plt.yscale('log')
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(['training loss','validation loss'],loc='best') # loc='best'
    plt.tight_layout()
    plt.grid()
    fig.savefig(dir_result+'loss.png')
    
    fig=plt.figure(2)
    plt.plot(cbk_ap.ap_train,'-*r')
    plt.plot(cbk_ap.ap_val,'-*b')
    #plt.yscale('log')
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('Average Precision',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(['training AP','validation AP'],loc='best') # loc='best'
    plt.tight_layout()
    plt.grid()
    plt.savefig(dir_result+'AP.png')
        
    np.save(dir_result+'ap_train.npy',cbk_ap.ap_train)
    np.save(dir_result+'ap_val.npy',cbk_ap.ap_val)

    with open(dir_result+'model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(dir_result+'model.h5')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_result', type=str, help='result directory')
    
    args = parser.parse_args()     
    main(args)

