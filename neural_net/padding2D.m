function [padded_img] = padding2D(input, pad)
% Effettua il padding di un'immagine
% input - Immagine dove mettere il padding
% pad - numero intero che identifica quanto padding introdurre
    if pad < 0
       throw(MException('myComponent:inputError', 'Padding non valido'));
    end
    dim = size(input);
    padded_img = zeros(dim(1)+2*pad, dim(2)+2*pad);
    padded_img_dim = size(padded_img);
    disp(pad+1);
    disp(dim(2)-pad);
    padded_img(pad+1:padded_img_dim(2)-pad, pad+1:padded_img_dim(1)-pad) = input;
end

