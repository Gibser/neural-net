c = 1;
for i=1 : 128 : 1024
    temp2(:, c) = [i i+127];
    c = c + 1;
end