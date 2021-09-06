clear
clc
close all

dir_data = fullfile('..','data');
dir_im_in = fullfile(dir_data, 'image864');
dir_lb_in = fullfile(dir_data, 'label864');
dir_msk_in = fullfile(dir_data, 'msk864');
dir_im_out = fullfile(dir_data, 'image864_rot');
dir_lb_out = fullfile(dir_data, 'label864_rot');
dir_msk_out = fullfile(dir_data, 'msk864_rot');

if ~exist(dir_im_out, 'dir')
    mkdir(dir_im_out);    
end
if ~exist(dir_lb_out, 'dir')
    mkdir(dir_lb_out);    
end
if ~exist(dir_msk_out, 'dir')
    mkdir(dir_msk_out);
end

d_ag=30;

info_im=dir([dir_im_in filesep '*.JPG']);
n_im=length(info_im);

for i=1:n_im
    im_name=info_im(i).name;
    im=imread([dir_im_in filesep im_name]);
    
    lb_name=[im_name(1:end-3) 'png'];
    lb=imread([dir_lb_in filesep lb_name]);
    
    msk_name=[im_name(1:end-3) 'png'];
    msk=imread([dir_msk_in filesep msk_name]);

    for ag=0: d_ag : 360-d_ag
         if ag==0
            im_r=im;
            lb_r=lb;
            msk_r=msk;
         else
            im_r = imrotate(im,ag,'bicubic','crop');
            lb_r =  imrotate(lb,ag,'bicubic','crop');
            msk_r =  imrotate(msk,ag,'bicubic','crop');
         end
        
        im_f=flip(im_r,2);
        lb_f=flip(lb_r,2);
        msk_f=flip(msk_r,2);
        
        f_im_r=[dir_im_out filesep im_name(1:end-4) '_' sprintf('%03d',ag) '_r.JPG'];
        f_im_f=[dir_im_out filesep im_name(1:end-4) '_' sprintf('%03d',ag) '_f.JPG'];
        
        f_lb_r=[dir_lb_out filesep im_name(1:end-4) '_' sprintf('%03d',ag) '_r.png'];
        f_lb_f=[dir_lb_out filesep im_name(1:end-4) '_' sprintf('%03d',ag) '_f.png'];
        
        f_msk_r=[dir_msk_out filesep im_name(1:end-4) '_' sprintf('%03d',ag) '_r.png'];
        f_msk_f=[dir_msk_out filesep im_name(1:end-4) '_' sprintf('%03d',ag) '_f.png'];
             
        imwrite(im_r, f_im_r);
        imwrite(im_f, f_im_f);
        
        imwrite(lb_r, f_lb_r);
        imwrite(lb_f, f_lb_f);
        
        imwrite(msk_r, f_msk_r);
        imwrite(msk_f, f_msk_f);
        
    end
    
end

