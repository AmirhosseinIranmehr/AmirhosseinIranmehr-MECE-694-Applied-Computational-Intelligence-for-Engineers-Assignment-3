% Accuracy data
Gaussian = [94.5, 91, 95.5];
Logistic = [95.5, 94, 96.5];
HyperbolicTangent = [88.5, 86.5, 90];
MATLABGaussian = [95, 95, 95.5];
MATLABPolynomial = [67.5, 58, 71];

% Creating a matrix of data
data = [Gaussian; Logistic; HyperbolicTangent; MATLABGaussian; MATLABPolynomial];

% Creating the bar chart
bar(data, 'grouped')
legend({'Problem 1', 'Problem 2', 'Problem 3'}, 'Location', 'best')
ylabel('Accuracy %')
xlabel('Kernel Type')
set(gca, 'xticklabel', {'Gaussian', 'Logistic', 'Hyperbolic Tangent', 'MATLAB Gaussian', 'MATLAB Polynomial'})
title('Comparison of SVM Kernel Accuracies')

% Problem sets
problems = 1:3;

