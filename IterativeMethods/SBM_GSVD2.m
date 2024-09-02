function[x,X,D,G] = SBM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,Ups,M,b,lambdaA,tau,tol,maxiter)
% Apply split Bregman to the l2-l1 problem where 
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
% tau: Shrinkage parameter
% tol: convergence tolerance
% maxiter: Maximum number of iteraitons
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors
% D: Matrix of d vectors
% G: Matrix of g vectors

if nargin < 6
    maxiter = 50;
end
if nargin < 5
    tol = 0.001;
end
if(length(lambdaA) == 1)
    lambdaA = ones(maxiter,1)*lambdaA;
end
n = sqrt(length(b));
p = size(V1,1);
x = zeros(n^2,1);
d = zeros(p^2,1);
g = d;
X = zeros(length(x),maxiter);
D = zeros(p^2,maxiter);
G = zeros(p^2,maxiter);
for i = 1:maxiter
    lambda= lambdaA(i,1);
    dg=d-g;
    % Solve the minimization for x
    Phi = Ups./(Ups.^2+lambda^2*M.^2);
    Psi = (lambda^2*M)./(Ups.^2+lambda^2*M.^2);
    xa = V2'*reshape(dg,p,p)*V1;
    xa = xa(1:n,1:n);
    x = X2*(reshape(Phi.*reshape(U2'*reshape(b,n,n)*U1,n^2,1)+Psi.*xa(:),n,n))*X1';
    x= x(:);
    X(:,i) = x;

    % Use shrinakge operators to solve for d
    %h = V2*[reshape(M.*reshape(X2i*reshape(x,n,n)*X1i',n^2,1),n,n) zeros(n,p-n);zeros(p-n,n) zeros(p-n,p-n)]*V1';
    h = V2*reshape(M.*reshape(X2i*reshape(x,n,n)*X1i',n^2,1),n,n)*V1';
    h = h(:);
    for j = 1:length(d)
        lim = h(j,1) + g(j,1);
        d(j,1) = sign(lim)*max(abs(lim)-tau,0);
    end
    D(:,i) = d;

    % Update g
    g = g + (h - d);
    G(:,i) = g;

    % Check if converged
    if i>1
        if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol && i>1
            X = X(:,1:i);
            D = D(:,1:i);
            G = G(:,1:i);
            break
        end
    end
end