function [X] = loadCatDogImages(folderpath1, folderpath2, total_images_num1, total_images_num2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    image_heigth = 100;
    image_width = 100;
    X = randn(image_heigth, image_width, total_images_num1+total_images_num2);
    j=0;
    for i=1:total_images_num1+1
        try
            IMG = imread(strcat(folderpath1, '\', num2str(j)  , '.jpg'));
            IMG = imresize(IMG,[image_heigth,image_width]);
            I = im2gray(IMG);
            I = normalize(I, 'range');
            X(:,:,i) = I;
        catch ERR
            disp(ERR);
        end
        j = j+1;
    end
    
    q=0;
    for k=(total_images_num1+1):total_images_num2 + total_images_num1 + 1
        %disp(num2str(i));
        try
            IMG = imread(strcat(folderpath2, '\', num2str(q), '.jpg'));
            IMG = imresize(IMG,[image_heigth,image_width]);
            I = im2gray(IMG);
            I = normalize(I, 'range');
            X(:,:,k) = I;
        catch ERR
            disp(ERR);
        end
        q = q+1;
       
    end
    
end

