function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z=X*theta; % theta transpose multiplied by X
H=sigmoid(z);
J= (sum(-y.*log(H)+y.*log(1-H)-log(1-H)) +(lambda/2)*sum(theta(2:length(theta),:).^2))./m;
%the second term is like that since theta j starts  from 1 and not 0
grad=(X'*(H-y))./m+(lambda/m).*theta;
%setting gradient for theta 0
grad(1)=(X'*(H-y))(1)./m;





% =============================================================

end
