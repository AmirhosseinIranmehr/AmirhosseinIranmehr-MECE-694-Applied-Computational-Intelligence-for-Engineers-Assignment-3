function [J, GradJ] = MyANNObjecFunction_Incomplete(ANN_Weights, ...
    NumFeatures, ...
    hiddenLayerSize, ...
    NumClasses, ...
    x, t, lambda)
% This function implements the "multi-layer perceptron" objective function
% and gradient for a two layer MLP.

%%% PHASE 0:
% MLP parameters are "unrolled" into the vector "ANN_Weights" and need
% to be converted back into the weight matrices.
% Reshape ANN_Parameters back into the parameters "theta1" and "theta2", the weight matrices
% for the two layer MLP
theta1 = reshape(ANN_Weights(1:hiddenLayerSize * (NumFeatures + 1)), ...
    hiddenLayerSize, (NumFeatures + 1));

theta2 = reshape(ANN_Weights((1 + (hiddenLayerSize * (NumFeatures + 1))):end), ...
    NumClasses, (hiddenLayerSize + 1));

% Number of Training Observations
M = size(x, 2);

% Complete the Code to Compute the Objective Value and Gradients
J = 0;
Grad_theta1 = zeros(size(theta1));
Grad_theta2 = zeros(size(theta2));

% ********************* Write You Code Here *********************
%%%% See the MLP Lecture Notes for Back-Propagation Algorithm

%%% PHASE 1: Forward Propagation
% Feedforward the MLP using given weights and return the objective value in J

% Calculate Output of First Layer
% Create Augmented x
x = [ones(1, size(x, 2)); x]; % !!!Complete This Line of Code!!!
% Calculate Weighted Inputs z2
z2 = theta1 * x ; % !!!Complete This Line of Code!!!
% Apply Activation Function to z2 to Get Hidden Layer Outputs a2
a2 = 1 ./ (1 + exp(-z2));  % !!!Complete This Line of Code!!!

% Calculate Output of Second Layer
% Repeat the Above Three Steps
a2 = [ones(1, size(a2, 2)); a2]; % !!!Complete This Line of Code!!!
z3 = theta2 * a2; % !!!Complete This Line of Code!!!
a3 = 1 ./ (1 + exp(-z3)); % !!!Complete This Line of Code!!!

% Assign a3 as the output of the network
f = a3;

% Calculate Logistic Regression Objective Value
J = 0;
for i = 1:M
    yi = t(:, i);  % target output for ith observation
    J = J + sum(yi .* log(f(:, i)) + (1 - yi) .* log(1 - f(:, i)));% !!!Complete This Line of Code!!!
end
J = -1/M * J;

% Add Regularization Term to Objective Function
 J = J + lambda / (2 * M) * (sum(sum(theta1(:, 2:end).^2)) + sum(sum(theta2(:, 2:end).^2))); % !!!Complete This Line of Code!!!

% PHASE 2: Back Propagation
% Implement Back Propagation to calculate the gradients Delta1 and Delta2.
Delta1 = zeros(size(theta1));
Delta2 = zeros(size(theta2));
for i = 1:M
    % Perform Forward Propagation to calculate layer outputs a_l
    % For Output of Hidden Layer
    a1 = x(:,i);
    z2 = theta1 * a1 ; % !!!Complete This Line of Code!!!
    a2 = [1; 1 ./ (1 + exp(-z2))]; % !!!Complete This Line of Code!!!
    
    % Calculate Output of Output Layer
    a2 = [1; 1 ./ (1 + exp(-z2))]; % !!!Complete This Line of Code!!!
    z3 = theta2*a2; % !!!Complete This Line of Code!!!
    a3 = 1 ./ (1 + exp(-z3)); % !!!Complete This Line of Code!!!
    
    % Calculate error in output later: error = network outupt - target
    delta3 = a3 - t(:, i); % !!!Complete This Line of Code!!!
    
    % Calculate error in 2nd (hidden) layer
    % Compute the sigmoid gradient for z2 of the current example
    sigmoidGradient_z2 = a2 .* (1 - a2);  % Sigmoid gradient for current z2
    delta2 = (transpose(theta2(:,2:end))*delta3).*sigmoidGradient_z2(2:end); % !!!Complete This Line of Code!!!
    
    % Update gradients to correct weights between input-hiddel layer (Delta1)
    % and hiddel-output layer (Delta2)
    Delta1 = Delta1 + delta2 * a1'; % !!!Complete This Line of Code!!!
    Delta2 = Delta2 + delta3 * a2'; % !!!Complete This Line of Code!!!
end
% Normalize Gradients by Number of Observations
Grad_theta1 = 1/M * Delta1;
Grad_theta2 = 1/M * Delta2;

% Add Regularization to Gradients
Grad_theta1(:,2:end) = Grad_theta1(:,2:end) + (lambda/M) * theta1(:,2:end); % !!!Complete This Line of Code!!!
Grad_theta2(:,2:end) = Grad_theta2(:,2:end) + (lambda/M) * theta2(:,2:end); % !!!Complete This Line of Code!!!

% *****************************************************************

% Unroll gradients
GradJ = [Grad_theta1(:) ; Grad_theta2(:)];

end
