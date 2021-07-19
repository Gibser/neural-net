function [L] = flatten_kernel(kernel)
    d=size(kernel);
    k_h=d(1);
    k_w=d(2);
    
    if length(d)==2
        d(3)=1;
    end
    if length(d)==4
        C_in=d(4);
    elseif length(d)==3
        C_in=1;
    end
    
    L = zeros(k_h*k_w*C_in, d(3));
    for i=1 :d(3)
        if length(d)==4
            arr=kernel(:,:,i,:);
        elseif length(d)==3
            arr=kernel(:,:,i);
        end
        L(:,i)=reshape(arr.',1,[])';
    end
end

