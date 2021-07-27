function [output_arr] = softmax_function(class_array)
    output_arr = exp(class_array) ./ sum(exp(class_array), 1);
end

