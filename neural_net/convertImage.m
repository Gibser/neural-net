function [I] = convertImage(filename)
RGB = imread(filename);
I = rgb2gray(RGB);
I = normalize(I, 'range');
colormap gray;
imagesc(I);
end
