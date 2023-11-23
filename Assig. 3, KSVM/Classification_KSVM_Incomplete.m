%% (1) Create Data
clear
clc
close all

% x: input data --> p by M (M: # of observations, p: # of features)
% Class 1
Mu1 = [5 5]; Sigma1 = [3  0.1; 0.1  3];
X1 = mvnrnd(Mu1, Sigma1, 50)';

Mu1 = [6 -2]; Sigma1 = [2  0; 0  2];
X1 = [X1 mvnrnd(Mu1, Sigma1, 50)'];

% Class 2
Mu2 = [1 1]; Sigma2 = [5  0.5; 0.5  5];
X2 = mvnrnd(Mu2, Sigma2, 100)';

x = [X1'; X2']';
t_org = [ones(size(X1,2),1); -1*ones(size(X2,2),1)]';

figure(1)
plot(X1(1,:), X1(2,:), 'b*', 'DisplayName','Class 1');
hold on
plot(X2(1,:), X2(2,:), 'r+', 'DisplayName','Class 2');
hold off
ylabel('x_2'), xlabel('x_1')
axis([-4 12 -4 11])
legend

clearvars -except x t_org

%% (2) Kernel Support Vectir Machine (KSVM), MATLAB
clc
% Define SVM Parameter
C = 2;

%%% Design SVM
SVM = fitcsvm(x', t_org', ...
    'boxconstraint', C, ...
    'KernelFunction', 'Gaussian', ... % Options are: gaussian, linear, polynomial
    'Crossval', 'off', ... % Options are: off, on
    'Solver', 'ISDA'); % Options are: ISDA, L1QP, SMO
    
%%% Test SVM
y = predict(SVM, x')';

% Creating proper output format for Confusion Matrix
t = [t_org; t_org];
t(1,t(1,:) == -1) = 0;
t(2,t(2,:) == 1) = 0; 
t(2,t(2,:) == -1) = 1;

y = [y; y];
y(1,y(1,:) == -1) = 0;
y(2,y(2,:) == 1) = 0; 
y(2,y(2,:) == -1) = 1;

% Plot
figure, plotconfusion(t, y, 'All Data');
title('Performance of MATLAB KSVM Implementation')

clearvars -except x t_org

%% (3) Kernel Support Vectir Machine (KSVM), Your Code
clc
% Define SVM Parameter
C = 10;

% STEP 0: Define Kernel Function
Kernel.Type = 'Gaussian';
Kernel.Parameter = 1;
K = KernelFunction_Incomplete(Kernel); % !!! Complete This Function !!!

%%% STEP 1: Calculate Matrix H = [h_ij]
% h_ij = y_i * y_j * Kernel(x_i, x_j)

NumObs = size(x,2);
H = zeros(NumObs, NumObs);
for i = 1:NumObs
    for j = i:NumObs
         H(i,j) = t_org(i) * t_org(j) * K(x(:,i), x(:,j));
         H(j,i) = H(i,j);
    end
end

% Define Vector of Minus Ones
 MinusOnes = -ones(NumObs,1); % !!! Complete This Function !!!

%%% STEP 2: Solve Quadratic Programming Problem
% Define Equality Constraints
Aeq = t_org;  beq = 0;  % !!! Complete This Function !!!
% Define Lower/Upper Bounds
lb = zeros(NumObs,1) ; ub = C * ones(NumObs,1); % !!! Complete This Function !!!

% Set Optimization Parameters
options = optimset('Algorithm', 'interior-point-convex', ...
    'Display', 'iter', 'MaxIter', 100);
% Solve Quadratic Programming
alpha = quadprog(H, MinusOnes, [], [], Aeq, beq, lb, ub, [], options)';

% Making Sure that Not Selected alphas are zero
alpha_AlmostZero = (abs(alpha) < max(abs(alpha))/1e5); % !!! Complete This Function !!!
alpha(alpha_AlmostZero) = 0;

%%% STEP 3: Find Support Vectors
 S = find( alpha > 0 & alpha < C) ; % !!! Complete This Function !!!

%%% STEP 4: Find Parameter Theta0
theta0 = 0;
for i = S
    sum_k = 0;
     for j = S
        sum_k = sum_k + alpha(j) * t_org(j) * K(x(:,j), x(:,i));
     end
     theta0 = theta0 + (t_org(i) - sum_k);
    % !!! Complete This Function !!!
end
theta0 = theta0/length(S);

t=t_org;
% Plot Nonlinear Decision Boundary
Curve = @(x1,x2)  MySumFunc([x1; x2], alpha(S), t(S), x(:,S), K) + theta0;
CurveA = @(x1,x2) MySumFunc([x1; x2], alpha(S), t(S), x(:,S), K) + theta0 + 1;
CurveB = @(x1,x2) MySumFunc([x1; x2], alpha(S), t(S), x(:,S), K) + theta0 - 1;

PlotRange = [min(x(1,:)) max(x(1,:)) min(x(2,:)) max(x(2,:))];
figure(1)
plot(x(1,t==1), x(2,t==1), 'b*', 'DisplayName','Class 1');
hold on
plot(x(1,t==-1), x(2,t==-1), 'r+', 'DisplayName','Class 2');
plot(x(1,S), x(2,S), 'ko', 'MarkerSize', 8,'DisplayName', 'Sup. Vect.');
fimplicit(Curve, PlotRange, 'Color', 'g', ...
    'LineWidth', 2, 'DisplayName', 'Dec. Boundary', 'MeshDensity', 50);
fimplicit(CurveA, PlotRange, 'Color', 'k', ...
    'LineWidth',1,'LineStyle',':', 'DisplayName', 'Sup. Vect.', 'MeshDensity', 50);
fimplicit(CurveB, PlotRange, 'Color', 'k', ...
    'LineWidth', 1, 'LineStyle', ':', 'DisplayName', 'Sup. Vect.', 'MeshDensity', 50);
hold off
ylabel('x_2'), xlabel('x_1')
axis([-4 12 -4 11])
legend('Location', 'northwest')

% Creating proper output format for Confusion Matrix
%t_org = t;
t = [t_org; t_org];
t(1,t(1,:) == -1) = 0;
t(2,t(2,:) == 1) = 0; 
t(2,t(2,:) == -1) = 1;

for i = 1:size(x,2)
    y(i) = sign(MySumFunc(x(:,i), alpha(S), t_org(S), x(:,S), K) + theta0);
end
y = [y; y];
y(1,y(1,:) == 1) = 1;
y(1,y(1,:) == -1) = 0;
y(2,y(2,:) == 1) = 0;
y(2,y(2,:) == -1) = 1; 

% Plot
figure, plotconfusion(t, y, 'All Data');
title('Performance of MATLAB KSVM Implementation')
t = t_org;

clearvars -except x t








