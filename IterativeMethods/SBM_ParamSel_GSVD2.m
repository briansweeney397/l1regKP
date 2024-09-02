function[x,X,D,G,LG,LStop] = SBM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsF,M,b,method,tau,tol,lamtol,maxiter,za)
% Apply split Bregman to the l2-l1 problem where A = kron(A1,A2) and 
% L = kron(L1,L2) have KP structure. The GSVDs of {A1,L1} and {A2,L2}
% are used to solve the problem and the parameter lambda is selected 
% at each iteration.
%
% Inputs:
% Note that Ups1, Ups2, M1, and M2 are vectors of upsilon and mu
% U1, U2, V1, V2, X1, X2, X1i, X2i, UpsF, M: GSVD matrices such that
    % A1 = U1*diag(Ups1)*X1i, L1 = V1*diag(M1)*X2i
    % A2 = U2*diag(Ups2)*X2i, L2 = V2*diag(M2)*X2i,
    % where UpsF = kron(Ups1, Ups2), M = kron(M1,M2),
    % X1 = inv(X1i), X2 = inv(X2i)
% b: Observed data b
% method: parameter selection method applied at each iteration
    % 'gcv': Use GCV at each iteraiton
    % 'cchi': Central chi^2 test
    % 'ncchi': Non-central chi^2 test where xbar = x^{(k)}
% tau: Shrinkage parameter
% tol: convergence tolerance
% lamtol: Tolerance on the relative change in lambda
% maxiter: Maximum number of iteraitons
% za: Critical value of z_(1-alpha/2) for the chi^2 tests (default: alpha =
% 0.95)
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors
% D: Matrix of d vectors
% G: Matrix of g vectors
% LG: Vector of lambda values selected
% LStop: Iteration when lamtol is satisfied and we stop selecting lambda

if nargin < 17
    za = 0.0627;
end
if nargin < 16
    maxiter = 50;
end
if nargin < 15
    lamtol = 0.01;
end
if nargin < 14
    tol = 0.001;
end

if strncmp(method,'gcv',3)
    sel = 1;
elseif strncmp(method,'cchi',4)
    sel = 2;
elseif strncmp(method,'ncchi',5)
    sel = 3;
else
    sel = 1;
end

LStop = 0;
p = size(V1,1);
n = size(U1,1);
x = zeros(n^2,1);
d = zeros(p^2,1);
g = d;
X = zeros(length(x),maxiter);
D = zeros(p^2,maxiter);
G = zeros(p^2,maxiter);
Ups = UpsF(1:n^2);
LG = zeros(maxiter,1);
lamcut = 0;
if sel == 1
    sm = [Ups,M];
end

% Start iterations
for i = 1:maxiter
    dg = d-g;

    % Select lambda with appropriate method
    if lamcut == 0
        if sel==1
            LG(i,1) = gcvIterGSVD2(U1,U2,V1,V2,sm,b,dg);
            conv = 1;
        elseif sel == 2
            vint = V2'*reshape(dg,p,p)*V1;
            xo = (1./M).*reshape(vint(1:n,1:n),n^2,1);
            xo = X2*reshape(xo(1:n^2),n,n)*X1';
            r = b - reshape(U2*reshape(Ups.*reshape(X2i*xo*X1i',n^2,1),n,n)*U1',n^2,1);
            [lg,conv] = ChiSqx0_GSVD2(U1,U2,r,UpsF,M,za,p^2);
            LG(i,1) = lg;
        elseif sel == 3
            vint = V2'*reshape(dg,p,p)*V1;
            xo = (1./M).*reshape(vint(1:n,1:n),n^2,1);
            xo = X2*reshape(xo(1:n^2),n,n)*X1';
            r = b - reshape(U2*reshape(Ups.*reshape(X2i*xo*X1i',n^2,1),n,n)*U1',n^2,1);
            s = U2'*reshape(r,n,n)*U1;
            s=s(:);
            q = Ups.*reshape(X2i*(reshape(x-xo(:),n,n))*X1i',n^2,1);
            [lg,~,conv] = ChiSqx0_noncentral_GSVD2(s,q,UpsF,M,za,p^2);
            LG(i,1) =lg;
        end
        lambdaO = LG(i,1);
    else
        LG(i,1) = LG(i-1,1);
    end

    % Check if RC(lambda) < TOL_lambda
    if i>1
        if abs(LG(i)^2 - LG(i-1)^2)/abs(LG(i-1)^2) < lamtol && lamcut == 0 && conv ==1
            lamcut= 1;
            LStop = i;
        end
    end

    % Find solution x to l2-l2 minimization
    Phi = Ups./(Ups.^2+lambdaO^2*M.^2);
    Psi = (lambdaO^2*M)./(Ups.^2+lambdaO^2*M.^2);
    xa = V2'*reshape(dg,p,p)*V1;
    xa = xa(1:n,1:n);
    x = X2*(reshape(Phi.*reshape(U2'*reshape(b,n,n)*U1,n^2,1)+Psi.*xa(:),n,n))*X1';
    x = x(:);
    X(:,i) = x;

    % Find d using shrinkage operators
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
        if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol
            X = X(:,1:i);
            D = D(:,1:i);
            G = G(:,1:i);
            LG = LG(1:i);
            if lamcut == 0
                LStop = i;
            end
            break
        end
    end
end
