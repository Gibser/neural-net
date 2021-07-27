function [I] = convertImage(filename)
% Funzione che prende in input un'immagine, la ridimensiona in 28x28 pixel,
% la esporta in grayscale e restituisce la matrice corrispondente che può
% essere usata per il training
% filename - la STRINGA contenente il path dell'immagine 
% I - restituisce la matrice corrispondente all'immagine grayscale
    RGB = imread(filename);
    RGB = imresize(RGB,[200,200]);
    I = rgb2gray(RGB);
    I = normalize(I, 'range');
    colormap gray;
    imagesc(I);
end
