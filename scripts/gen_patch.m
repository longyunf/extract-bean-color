clear
clc
close all

dir_lib = fullfile('..','lib');
addpath(dir_lib)

dir_data = fullfile('..','data');
dir_im_in = fullfile(dir_data, 'image864_rot');
dir_lb_in = fullfile(dir_data, 'label864_rot');
dir_msk_in = fullfile(dir_data, 'msk864_rot');
dir_im_out = fullfile(dir_data, 'image_patch');
dir_lb_out = fullfile(dir_data, 'label_patch');
dir_msk_out = fullfile(dir_data, 'msk_patch');

if ~exist(dir_im_out, 'dir')
    mkdir(dir_im_out);    
end
if ~exist(dir_lb_out, 'dir')
    mkdir(dir_lb_out);    
end
if ~exist(dir_msk_out, 'dir')
    mkdir(dir_msk_out);
end

info_im=dir([dir_im_in filesep '*.JPG']);
n_im=length(info_im);
p_size=128;  

for i=1:n_im
    im_name=info_im(i).name;
    im=imread([dir_im_in filesep im_name]);
    
    lb_name=[im_name(1:end-3) 'png'];
    lb=imread([dir_lb_in filesep lb_name]);
    
    msk_name=[im_name(1:end-3) 'png'];
    msk=imread([dir_msk_in filesep msk_name]);
    
    c_im=im2patch(im, p_size);
    c_lb=im2patch(lb, p_size);
    c_msk=im2patch(msk, p_size);
    
    n=length(c_im);
    
    for j=1:n
        f_im_out=[dir_im_out filesep im_name(1:end-4) '_' sprintf('%03d',j) '.JPG'];
        f_lb_out=[dir_lb_out filesep im_name(1:end-4) '_' sprintf('%03d',j) '.png'];
        f_msk_out=[dir_msk_out filesep im_name(1:end-4) '_' sprintf('%03d',j) '.png'];
        imwrite(c_im{j}, f_im_out);
        imwrite(c_lb{j}, f_lb_out);
        imwrite(c_msk{j}, f_msk_out); 
    end
       
end
