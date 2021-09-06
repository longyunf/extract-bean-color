clear
clc
close all

dir_data = fullfile('..','data');

dir_train=[dir_data filesep 'train_patch'];
dir_val=[dir_data filesep 'val_patch'];

dir_im_rot=[dir_data filesep 'image_patch'];
dir_im_train=[dir_train filesep 'image_rot'];

dir_label_rot=[dir_data filesep 'label_patch'];
dir_label_train=[dir_train filesep 'label_rot'];
dir_msk_rot=[dir_data filesep 'msk_patch'];
dir_msk_train=[dir_train filesep 'msk_rot'];

dir_im_train2=[dir_train filesep 'image'];
dir_label_train2=[dir_train filesep 'label'];
dir_msk_train2=[dir_train filesep 'msk'];

dir_im_val=[dir_val filesep 'image'];
dir_label_val=[dir_val filesep 'label'];
dir_msk_val=[dir_val filesep 'msk'];

dir_im=[dir_data filesep 'image864'];
info_im=dir([dir_im filesep '*.JPG']);

idx_full=1:24;
idx_val=[9, 16, 18:20, 22, 24];
idx_train=setdiff(idx_full, idx_val);

% train
for i=1:length(idx_train)
    idx=idx_train(i);
    name=info_im(idx).name;
    name_common=name(1:end-4);
    
    status = copyfile([dir_im_rot filesep name_common '*'], dir_im_train); 
    disp(status);
    status = copyfile([dir_label_rot filesep name_common '*'], dir_label_train); 
    disp(status);
    status = copyfile([dir_msk_rot filesep name_common '*'], dir_msk_train); 
    disp(status);
    
    status = copyfile([dir_im_rot filesep name_common '_000_r*'], dir_im_train2);
    disp(status);
    status = copyfile([dir_label_rot filesep name_common '_000_r*'], dir_label_train2); 
    disp(status);
    status = copyfile([dir_msk_rot filesep name_common '_000_r*'], dir_msk_train2); 
    disp(status);      
end

% validation
for i=1:length(idx_val)
    idx=idx_val(i);
    name=info_im(idx).name;
    name_common=name(1:end-4);
    
    status = copyfile([dir_im_rot filesep name_common '_000_r*'], dir_im_val); 
    disp(status);
    status = copyfile([dir_label_rot filesep name_common '_000_r*'], dir_label_val); 
    disp(status);
    status = copyfile([dir_msk_rot filesep name_common '_000_r*'], dir_msk_val); 
    disp(status);
end

