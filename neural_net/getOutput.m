function [res] = getOutput(pred)
    arr = softmax_function(pred);
    p = max(arr)*100;
    index = interp1(arr,1:numel(arr),max(arr))-1;
    arr(index+1) = 0;
    p2 = max(arr)*100;
    index2 = interp1(arr,1:numel(arr),max(arr))-1;
    disp(['Result:', num2str(index)]);
    disp(['Probability:', num2str(p), '%']);
    disp(['Second best match:',  num2str(index2)]);
    disp(['Probability:',  num2str(p2), '%']);
    disp(softmax_function(pred)');
end

