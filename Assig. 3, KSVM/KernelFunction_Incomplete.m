function [K] = KernelFunction_Incomplete(Kernel)
% This function retuns function handle to various kernel functions to 
% map data to a higher dimension 

% Kernel.Type: Name of the kernel function including Linear, Gaussian,
% Logistic, HypTan

% Kernel.Parameter: Parameter(s) associated with each kernel function

% K: function handle for a user specified kernel function
% K operates on two inputs xi and xj

% Provide the code to compute Gaussian, Logistic, HypTan kernels
% Linear kernel function is provided as an example for you

% ************** Write Your Code Here **********************
switch Kernel.Type
    case 'Linear'
        K = @(xi, xj) (1+xi'*xj)^Kernel.Parameter;
    case 'Gaussian'
        K = @(xi, xj) exp(-1/(2*Kernel.Parameter^2) * norm(xi - xj)^2);
    case 'Logistic'
        K = @(xi, xj) 1 / (1 + exp(-Kernel.Parameter * norm(xi - xj)^2));
    case 'HypTan'
        beta0= -2; % Offset Parameter
        K = @(xi, xj) tanh(beta0 + Kernel.Parameter * (xi'*xj));
        
% **********************************************************
end