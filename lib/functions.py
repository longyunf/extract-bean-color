import numpy as np
import glob
import skimage.io as io
from sklearn.metrics import average_precision_score  

from tensorflow import keras 
import tensorflow as tf                                                                                                                                                                     


def loss(y_true, y_pred):
    msk=y_true[:,:,2]
    n_pixel=tf.reduce_sum(msk)
    wt=tf.reduce_sum([y_true[:,:,0] * msk])/n_pixel
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    loss=tf.reduce_sum( (1-wt) * y_true[:,:,0] * (-tf.math.log(y_pred[:,:,0])) * msk + wt * y_true[:,:,1] * (-tf.math.log(y_pred[:,:,1])) * msk) / n_pixel
    return loss


def decomposeImage(im, patch_size):
    # decompose an image into patches
    # input:
    # im: h x w x 3
    # output:
    # patches: n_patch x h x w x3
    
    (h_im, w_im, _)=im.shape
    n_h = h_im//patch_size + 1
    n_w = w_im//patch_size + 1
    N=n_h*n_w
    
    h_pad=n_h*patch_size-h_im
    w_pad=n_w*patch_size-w_im
    
    im_temp=np.pad(im, ((0,h_pad),(0,w_pad),(0,0)), 'constant')
    
    patches=np.zeros((N, patch_size, patch_size, 3))
    ct=0
    for i in range(n_h):
        for j in range(n_w):
            patches[ct,]=im_temp[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]
            ct+=1
    
    return patches


def synthesizePatch(prd_patches, h_im, w_im):
    # synthesize predicted patches
    # input:
    # prd_patches: N x patch_size x patch_size
    # h_im, w_im: height and width of a full image
    # output:
    # prd_full: h_im x w_im
    
    patch_size=prd_patches.shape[-1]
    n_h = h_im//patch_size + 1
    n_w = w_im//patch_size + 1
    
    prd_patches=prd_patches.reshape((n_h, n_w, patch_size, patch_size))
    
    prd_full=np.zeros((n_h*patch_size, n_w*patch_size))
    
    for i in range(n_h):
        for j in range(n_w):
            prd_full[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]=prd_patches[i,j,...]
            
    prd_full=prd_full[:h_im,:w_im]
    
    return prd_full
            
            
def prd_1_image(im, model):    
    (h,w,_)=im.shape

    x=im[np.newaxis,...]	
    prb_hilum = model.predict(x)  
    prb_hilum=prb_hilum[...,1].reshape(h, w)
    
    return prb_hilum


def prd_1_image_patch(im, model, patch_size):
    # predict 1 image by decomposition and synthesis of patches
    # output:
    # prb_hilum: shape (h, w)
    
    (h_im,w_im,_)=im.shape
    patches=decomposeImage(im, patch_size)
    y_prd=prd_image_by_image(model, patches)
    n_patch=patches.shape[0]
    y_prd=y_prd[...,1].reshape((n_patch, patch_size, patch_size))
    prb_hilum=synthesizePatch(y_prd, h_im, w_im)
    return prb_hilum


def load_input_image(dir_im, im_shape):
    # inputs:
    # dir_im: image input directory
    # im_shape: (h,w,n_channel)
    # output:
    # x: numpy array of shape (n_sample, h, w, n_channels)
    im_names=np.array(np.sort(glob.glob(dir_im+'*.[Jj][Pp][Gg]')))
    n_sample=len(im_names)
    
    x=np.empty((n_sample,)+im_shape) 
    for i, path_im in enumerate(im_names):
        x[i,]=io.imread(path_im)   
    return x


def load_label(dir_label, label_shape):
    # inputs:
    # dir_label: label directory
    # label_shape: (h,w)
    # output:
    # y: numpy array of shape (n_sample, h, w)
    f_names=np.array(np.sort(glob.glob(dir_label+'*.png')))
    n_sample=len(f_names)    
    
    y=np.empty((n_sample,)+label_shape) 
    for i, path_f in enumerate(f_names):
        y[i,]=io.imread(path_f)      
    y[y==255]=1
    return y


def prd_image_by_image(model, x):
    # predict images one by one
    # input
    # x: (n_sample, h, w, 3)
    # ouput:
    # y_prd: (n_sample, h*w, 2)
    n_sample=x.shape[0]
    h=x.shape[1]
    w=x.shape[2]

    temp=np.empty((1,h,w,3))
    y_prd=np.empty((n_sample,h*w,2))
  
    for i in range(n_sample):
        temp[0]=x[i]
        y_prd[i]=model.predict(temp)    #1 x h*w x 2
    
    return y_prd


# compute average precision
class cbk_comp_ap(keras.callbacks.Callback):
  def __init__(self, dir_im_train, dir_im_val, dir_label_train, dir_label_val, dir_msk_train, dir_msk_val, class_id):
      self.ap_train = []
      self.ap_val = []
      self.dir_im_train=dir_im_train
      self.dir_im_val=dir_im_val
      self.dir_label_train=dir_label_train
      self.dir_label_val=dir_label_val
      self.dir_msk_train=dir_msk_train
      self.dir_msk_val=dir_msk_val
      self.class_id=class_id
 
  def on_epoch_end(self, epoch, logs={}): 
      h=128
      w=128      
      
      # training data      
      x_train=load_input_image(self.dir_im_train, (h,w,3))
      y_train=load_label(self.dir_label_train, (h,w))
            
      y_cls=y_train==self.class_id
      y_cls=np.reshape(y_cls,(-1))
      
      msk=load_label(self.dir_msk_train, (h,w))
      msk[msk==0]=tf.keras.backend.epsilon()
      msk=np.reshape(msk,(-1))
      
      y_train_prd = prd_image_by_image(self.model, x_train)
      score=y_train_prd[...,self.class_id]
      score=np.reshape(score,(-1))

      ap=average_precision_score(y_cls , score, sample_weight=msk)
      print('Training AP: %.4f'%(ap))    
      self.ap_train.append(ap)
           
      # validation data
      x_val=load_input_image(self.dir_im_val, (h,w,3))
      y_val=load_label(self.dir_label_val, (h,w))
     
      y_cls=y_val==self.class_id
      y_cls=np.reshape(y_cls,(-1))
      
      msk=load_label(self.dir_msk_val, (h,w))
      msk[msk==0]=tf.keras.backend.epsilon()
      msk=np.reshape(msk,(-1))
      
      y_val_prd = prd_image_by_image(self.model, x_val)
      score=y_val_prd[...,self.class_id]
      score=np.reshape(score,(-1))

      ap=average_precision_score(y_cls, score, sample_weight=msk)
      print('Validation AP: %.4f'%(ap))   
      self.ap_val.append(ap)

