function c=im2patch(im, p_size)
% generate image patches
% input: im: h x w x n or h x w
% output:
% c: cell array (1 x n), n patches

step=floor(p_size/8)*7;
[h, w, ~]=size(im);

n_pad_h=ceil((h-p_size)/step)*step+p_size-h;
n_pad_w=ceil((w-p_size)/step)*step+p_size-w;

im = padarray(im, [n_pad_h, n_pad_w], 0, 'post');

[h2, w2, ~]=size(im);

n_w=(w2-p_size)/step+1;
n_h=(h2-p_size)/step+1;

c=cell(n_h,n_w);

for i=1:n_h
    for j=1:n_w
        c{i,j}=im(step*(i-1)+1: step*(i-1)+p_size, step*(j-1)+1: step*(j-1)+p_size, :);
    end 
end

c=reshape(c, [1, n_h*n_w]);

