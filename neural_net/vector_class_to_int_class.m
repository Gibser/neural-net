function [class] = vector_class_to_int_class(vector)
    %Restituisce la classe a partire da un array one-hot
    %es. [0 0 0 1 0 0] -> 3
    [~, class] = max(vector, [], 2);
    class = class - 1;
end

