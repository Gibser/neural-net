function [M] = flatten_input(input, k_h, k_w, s, P)
% Flatterns the input based on kernel's size
% input - input matrix
% k_h   - kernel's heigth
% k_w   - kernel's width
% s     - stride
% P     - padding

    d = size(input);
    if length(d) == 2       %Se l'input è un sola immagine imposto batchsize a 1
        d(3) = 1;
    end
    
    if length(d) == 4
        C_in = d(4);
    elseif length(d) == 3
        C_in = 1;
    end
    
    H_out = (d(1) + 2*P - k_h)/s + 1;
    W_out = (d(2) + 2*P - k_w)/s + 1;
    M = zeros(d(3)*H_out*W_out, k_h*k_w*C_in);
    for j=0 : d(3)*H_out*W_out-1
        l = floor(j / (H_out*W_out));
        p = mod(j, H_out*W_out);
        m = floor(p/H_out);
        t = mod(p, W_out);
        isw = t*s;
        ish = m*s;
        if length(d) == 4
            arr = input(ish+1:ish+k_h, isw+1:isw+k_w, l+1, :);
        else
            arr = input(ish+1:ish+k_h, isw+1:isw+k_w, l+1);
        end
        M(j+1, :) = reshape(arr.',1,[]);
    end
end

