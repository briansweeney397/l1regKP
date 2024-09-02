function[x,X,H] = MM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,Ups,M,b,lambdaA,epsilon,tol,maxiter)
% Apply Majorization-Minimization to the l2-l1 problem where 
% A = kron(A1,A2) and L = kron(L1,L2) have KP structure. 
% The GSVDs of {A1,L1} and {A2,L2} are used to solve the problem.
%
% Inputs:
% Note that Ups1, Ups2, M1, and M2 are vectors of upsilon and mu
% U1, U2, V1, V2, X1, X2, X1i, X2i, UpsF, M: GSVD matrices such that
    % A1 = U1*diag(Ups1)*X1i, L1 = V1*diag(M1)*X2i
    % A2 = U2*diag(Ups2)*X2i, L2 = V2*diag(M2)*X2i,
    % where UpsF = kron(Ups1, Ups2), M = kron(M1,M2),
    % X1 = inv(X1i), X2 = inv(X2i)
% b: Observed data b
% lambdaA: Value (or vector of values) of lambda to use at each iteration
% epsilon: Smoothing parameter
% tol: convergence tolerance
% maxiter: Maximum number of iteraitons
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors
% H: Matrix of h vectors

if nargin < 15
    maxiter = 50;
end
if nargin < 14
    tol = 0.001;
end
if(length(lambdaA) == 1)
    lambdaA = ones(maxiter,1)*lambdaA;
end
n = sqrt(length(b));
p = size(V1,1);
x0 = zeros(n^2,1);
x =x0;
X = zeros(length(x0),maxiter);
H = zeros(p^2,maxiter);
for i = 1:maxiter
    lambda = lambdaA(i,1);
    %h = V2*[reshape(M.*reshape(X2i*reshape(x,n,n)*X1i',n^2,1),n,n) zeros(n,p-n);zeros(p-n,n) zeros(p-n,p-n)]*V1';
    h = V2*reshape(M.*reshape(X2i*reshape(x,n,n)*X1i',n^2,1),n,n)*V1';
    u = h(:);
    wreg = u.*(1-((u.^2+epsilon^2)./epsilon^2).^(1/2-1));
    H(:,i) = wreg;
    % Solve minimization for x
    Phi = Ups./(Ups.^2+lambda^2*M.^2);
    Psi = (lambda^2*M)./(Ups.^2+lambda^2*M.^2);
    xa = V2'*reshape(wreg,p,p)*V1;
    xa = xa(1:n,1:n);
    x = X2*(reshape(Phi.*reshape(U2'*reshape(b,n,n)*U1,n^2,1)+Psi.*xa(:),n,n))*X1';
    x= x(:);
    X(:,i) = x;

    % Check if converged
    if i>1
        if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol && i>1
            X = X(:,1:i);
            H = H(:,1:i);
            break
        end
    end
end