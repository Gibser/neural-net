function [X, Y] = shuffleData(X,Y)
   
    ix = randperm(size(Y,1));
    
    Y = Y(ix,:);
    X = X(:,:,ix);
   
    
end

