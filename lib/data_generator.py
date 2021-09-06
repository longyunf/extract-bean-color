import numpy as np
import glob
import os
import skimage.io as io
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dir_input, dir_label, dir_msk, batch_size=2, dim_input=(864,864,3), 
                 dim_label=(864,864,1), shuffle=False):                      
        self.dir_input=dir_input
        self.dir_label=dir_label
        self.dir_msk=dir_msk
        self.batch_size = batch_size
        self.dim_input = dim_input
        self.dim_label = dim_label
        self.shuffle = shuffle
        self.im_names=np.array(np.sort(glob.glob(self.dir_input+'*.JPG')))
        self.indexes = np.arange(len(self.im_names))
        self.list_IDs = np.arange(len(self.im_names))

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_batch = [self.list_IDs[k] for k in indexes_batch]
        x, y = self.__data_generation(list_IDs_batch)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
                   
    def __data_generation(self, list_IDs_batch):
        x = np.empty((self.batch_size, *self.dim_input))
        y = np.empty((self.batch_size, *self.dim_label))
        msk = np.empty((self.batch_size, *self.dim_label))
             
        for i, ID in enumerate(list_IDs_batch):
            path_im=self.im_names[ID]
            im = io.imread(path_im)                 
            x[i,]=im.copy()    
            
            im_name=os.path.basename(path_im)
            path_lb=self.dir_label+im_name[0:-3]+'png'
            path_msk=self.dir_msk+im_name[0:-3]+'png'
            
            lb = io.imread(path_lb)
            msk_temp = io.imread(path_msk)
            
            y[i,]=lb[..., np.newaxis].copy()
            msk[i,]=msk_temp[..., np.newaxis].copy()
        
        y[y==255]=1
        msk[msk==255]=1

        y = keras.utils.to_categorical(y, 2)
        
        y=np.reshape(y,(self.batch_size,-1,2))
        msk=np.reshape(msk,(self.batch_size,-1,1))
        y=np.concatenate((y, msk), axis=2)

        return x, y
    