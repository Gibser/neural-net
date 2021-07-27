function [new_labels] = build_Y(labels)
%%Restituisce la one-hot encoding di un insieme di labels per MNIST
% labels - label del dataset MNIST
% new_labels - label codificati tramite la one-hot
   new_labels= zeros(size(labels,1),2);
   for i=1 :size(labels,1)
      if(labels(i)) == 1
        new_labels(i, 1) = 1; 
      else
         new_labels(i, 2) = 1; 
      end
   end
end

