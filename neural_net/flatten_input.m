function [M] = flatten_input(input, kernel, k_h, k_w, s, P)
    d = size(input);
    dim_k = size(kernel);
    if length(d) == 4
        C_in = d(4);
    else
        C_in = 1;
    end
    H_out = (d(1) + 2*P - k_h)/s + 1;
    W_out = (d(2) + 2*P - k_w)/s + 1;
    M = zeros(d(3)*H_out*W_out, k_h*k_w*C_in);
    L = zeros(k_h*k_w*C_in, d(3));
    
    for j=1 : d(3)*H_out*W_out
        l = floor(j / (H_out*W_out)) + 1;
        p = mod(j, H_out*W_out);
        m = floor(p/H_out) + 1;
        t = mod(p, W_out);
        isw = t*s;
        ish = m*s;
        disp(isw);
        disp(ish);
        disp(l);
        if length(d) == 4
            arr = input(ish:ish+s, isw:isw+s, l, :);
        else
            arr = input(ish:ish+s, isw:isw+s, l);
        end
        
        M(j, :) = arr(:);
    end
end
