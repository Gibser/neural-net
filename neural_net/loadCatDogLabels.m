function [Y] = loadCatDogLabels(total_images_num1, total_images_num2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    Y = randn(total_images_num1+total_images_num2,1);
  
    for i=1:total_images_num2 + total_images_num1 + 1
        if i <= total_images_num1
           Y(i) = 1;
        else
            Y(i) = 0;
        end
    end 
end



