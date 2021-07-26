function getOutput(pred)
    arr = softmax_function(pred);
    [p, index] = max(arr);
    arr(index) = 0;
    [p2, index2] = max(arr);
    disp(['Result:', num2str(index-1)]);
    disp(['Probability:', num2str(p*100), '%']);
    disp(['Second best match:',  num2str(index2-1)]);
    disp(['Probability:',  num2str(p2*100), '%']);
    disp(softmax_function(pred)');
end

