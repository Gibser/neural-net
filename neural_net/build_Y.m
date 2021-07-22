function [new_YT] = build_Y(YT)
   new_YT= zeros(size(YT,1),10)-1;
   for i=1 :size(YT,1)
      new_YT(i, YT(i)+1 ) = 1; 
   end
end

