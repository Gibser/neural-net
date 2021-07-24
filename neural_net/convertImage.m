function [I] = convertImage(filename)
RGB = imread(filename);
imshow(RGB)
I = rgb2gray(RGB);
figure
I = normalize(I, 'range');
colormap gray;
imagesc(I);

end
