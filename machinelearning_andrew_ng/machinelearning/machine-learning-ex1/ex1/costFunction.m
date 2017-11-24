function [jVal, gradient] = costFunction(theta)

data = load('ex1data2.txt');
X = data(:, 1:2); y = data(:, 2);
theta = zeros(2, 1);
H=X*theta;
  jVal = sum((H-y).^2)./(2*length(y)); %[...code to compute J(theta)...];
  gradient = (X'*(H-y)); %[...code to compute derivative of J(theta)...];
end
