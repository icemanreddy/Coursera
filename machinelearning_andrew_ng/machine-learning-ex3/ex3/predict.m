function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%X is a 5000 X 400 matrix.Theta1 is a 25 X 401 matrix
X = [ones(m, 1) X]; % adding bias input
z=X*Theta1';
a1=sigmoid(z); %5000 X 25 matrix
% adding the  bias input to a1. a1 is the input for the next layer
a1=[ones(size(a1,1), 1) a1];
%a1 is now a 5000 X 26 matrix .Theta2 is a 26 X 10 matrix
a2=sigmoid(a1*Theta2');
% a2 is 5000 X 10 matrix
% r contains the highest value in each row,p contains the column number of the highest value.
[r,p]=max(a2,[],2);







% =========================================================================


end
