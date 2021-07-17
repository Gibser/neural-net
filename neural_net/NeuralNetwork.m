classdef NeuralNetwork < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        weights
        biases
        activations
        deriv_func
        n_neurons
        n_layers
        weights_prev
    end
    
    methods
        function obj = NeuralNetwork(n_neurons, actv_functions, deriv_func, n_layers)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            SIGMA = 0.1;
            for i=2 : n_layers
                obj.weights{i-1} = SIGMA*randn(n_neurons(i), n_neurons(i-1));
                obj.biases{i-1} = SIGMA*randn(n_neurons(i), 1);
                obj.activations{i-1} = actv_functions{i-1};
                obj.deriv_func{i-1} = deriv_func{i-1};
            end
            obj.n_neurons = n_neurons;
            obj.n_layers = n_layers;
        end
        
        function [a_, z_] = forward_step(obj, x)
           z = x;
           for i=1 : obj.n_layers-1
               a = obj.weights{i} * z + obj.biases{i};
               %disp(a);
               a_{i} = a;
               z = obj.activations{i}(a);
               z_{i} = z;
           end
        end
        
        function [W_deriv, bias_deriv] = backpropagation(obj, x, t, derivFunErr)
            W_deriv = {};
            deltas = {};
            bias_deriv = {};
            %% FASE FORWARD-PROPAGATION
            [a_, z_] = obj.forward_step(x);
            z_ = [{x} z_(:)'];

            %% FASE BACK-PROPAGATION (calcolo delta)
            %Calcolo dela nodi di uscita
            delta_out = obj.deriv_func{end}(a_{end});
            delta_out = delta_out .* derivFunErr(z_{end}, t);
            deltas{obj.n_layers-1} = delta_out;
            W_deriv{obj.n_layers-1} = delta_out * z_{end-1}';
            %disp(W_deriv);
            w = 0; %Questo indice serve per iterare sulle matrici W della rete
            a = 1; %Questo indice serve per gli input a dei neuroni nei livelli
            z = 2;
            bias_deriv{obj.n_layers-1} = sum(delta_out, 2);
            for i=obj.n_layers-2 : -1: 1
                %disp(deltas{i+1});
                deltas{i} = obj.weights{end-w}' * deltas{i+1};
                deltas{i} = deltas{i} .* obj.deriv_func{end-a}(a_{end-a});
                W_deriv{i} = deltas{i} * z_{end-z}';
                bias_deriv{i} = sum(deltas{i}, 2);
                w = w + 1;
                a = a + 1;
                z = z + 1;
                %disp(W_deriv);
            end
        end
        
        function gradientDescent(obj, eta, W_deriv, bias_deriv)
            obj.weights_prev = obj.weights(:, :);
            for i=1 : obj.n_layers-1
                obj.weights{i} = obj.weights{i} - eta*W_deriv{i};
                obj.biases{i} = obj.biases{i} - eta*bias_deriv{i};
                %disp(obj.weights)
            end
        end
        
        function [err, err_val] = train(obj, N, x, t, x_val, t_val, errFunc, errFuncDeriv, eta, BATCH)
         %Learning rate e' un parametro che posso scegliere, e relativo alla regola di aggiornamento
         % scelta, ed è considerato un iper-parametro del processo di
         % learning. In genere passato come parametro alla funzione.
         %Apprendimento Batch (1) oppure on-line (0). In genere come parametro della funzione
         err = zeros(1,N);
         err_val = zeros(1,N);
         [~, z_] = obj.forward_step(x_val);
         y_val = z_{end};
         min_err = errFunc(y_val, t_val);%sumOfSquares(y_val, t_val)

         for epoch=1:N %In QUESTO CASO sto suppenendo di fare sempre tutte le iterazioni
             %LEARNING ON-LINE
             if BATCH==0
                 for n=1:size(x,2)
                     [w, b] = obj.backpropagation(x(:,n), t(:,n), errFuncDeriv);
                     %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
                     obj.gradientDescent(eta, w, b);
                 end
             else
              %BATCH LEARNIG
              [w, b] = obj.backpropagation(x, t, errFuncDeriv);
              %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
              obj.gradientDescent(eta, w, b);
             end

             [~, z2_] = obj.forward_step(x);
             [~, z_] = obj.forward_step(x_val);
             y = z2_{end};
             y_val = z_{end};
             err(epoch) = errFunc(y,t); %Cross-entropy for multiple classes
             %err(epoch)=sumOfSquares(y,t);
             %err_val(epoch)=sumOfSquares(y_val,t_val);
             err_val(epoch) = errFunc(y_val,t_val);
             disp(['err train:' num2str(err(epoch)) ' err val:' num2str(err_val(epoch))]);
             if err_val(epoch)< min_err
                 min_err=err_val(epoch);
             else
                 obj.weights = obj.weights_prev(:, :);
             end
          end
        end
    end
end

