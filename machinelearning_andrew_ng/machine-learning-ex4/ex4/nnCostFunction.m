function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Add the ones to the start of a1!
a1=[ones(m,1),X];   % 5000x401
z2=a1*Theta1';      % 5000x25
a2=sigmoid(z2);     % 5000x25
a2_=[ones(m,1),a2]; % 5000x26
z3=a2_*Theta2';  %5000x10
h=sigmoid(z3); %5000x10
y_=zeros(m,num_labels);
one = ones(m, num_labels);
for iter_i=1:size(y,1)
  y_(iter_i,y(iter_i))=1;
endfor
J=0;
t1= y_;
t2=log(h);
t3=1-y_;
t4=log(1-h);
J=sum(sum((-t1.*t2).-(t3.*t4)))/m;

%theta1 without the theta for the bias i.e col is to be skipped
Theta1_=Theta1(:,2:size(Theta1,2));
Theta2_=Theta2(:,2:size(Theta2,2));
J=J+(sum(sum((Theta1_.^2)))+sum(sum(Theta2_.^2)))*lambda/(2*m);



d3=h-y_; % 5000x10
d2=(d3*Theta2_).*sigmoidGradient(z2); % 5000x25
D1=d2'*a1;
D2=d3'*a2_;

%D1=0;
%D2=0;
%for t=1:m
%  d3=h(t,:)-y_(t,:);
%  d2=(d3*Theta2_).*sigmoidGradient(z2(t,:));
%  D1=D1+d2'*a1(t,:);
%  D2=D2+d3'*a2_(t,:);
%endfor

%step 5
Theta1_grad=D1/m;
Theta2_grad=D2/m;

%Regularized NN
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*Theta1_;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*Theta2_;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
