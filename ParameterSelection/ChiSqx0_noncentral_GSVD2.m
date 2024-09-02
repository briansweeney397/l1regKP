function [lambda,tolB,conv] = ChiSqx0_noncentral_GSVD2(s,q,ups,M,za,p)
% Use the non-central Chi^2 test to select lambda using the GSVD
%
% Inputs:
% s: Value of s in the chi^2 test (U'*(b-A*xo))
% q: Value of q in the chi^2 test (U'*(A*x-A*x0))
% ups, M: vectors of the upsilon and mu values from the GSVD
% za: Critical value of z_(1-alpha/2)
% p: Number of rows in L
%
% Outputs:
% lambda: Regularization parameter selected by chi^2 test
% tolB: Vector of bounds for Newton's method by iteration
% conv: indicator of whether the method converged with Newton

m = length(ups);
n=m;
lambda = 1;
gamma = ups(1:n)./M;

mt = m-n+min(n,p) - sum(s(n+1:m).^2-q(n+1:m).^2);
s = s(1:n);
q = q(1:n);
z = s.^2-q.^2;
zt1 = z./((gamma.^2+lambda^2));
zt2 = z./((gamma.^2+lambda^2).^2);
nc = lambda^2*sum((q.^2)./((gamma.^2+lambda^2)));
tolB = zeros(50,1);

f = sum(lambda^2*zt1)-mt;
dof = m-n + min(n,p);

% Newton's Method
iter = 0; 
while abs(f) > sqrt(2*(dof+2*nc))*za && iter < 50
    tolB(iter+1,1) = sqrt(2*(dof+2*nc))*za;
fp = 2*lambda*sum(zt2.*gamma.^2);
lambda = lambda-f/fp;
zt1= z./((gamma.^2+lambda^2));
zt2= zt1./((gamma.^2+lambda^2));
f = sum(lambda^2*zt1)-mt;
nc = lambda^2*sum((q.^2)./((gamma.^2+lambda^2)));
iter = iter + 1;
end
conv = 1;
tolB = tolB(1:iter);

% If no root, search for minimum derivative
if iter ==50 || abs(lambda) >1e4
    conv = 0;
    lambda = 1;
zt1= z./((gamma.^2+lambda^2));
zt2= zt1./((gamma.^2+lambda^2));
f = 2*lambda*sum(zt2.*gamma.^2);
    iter = 0; 
while abs(f) > sqrt(2*(dof))*za && iter < 50
    fp = 2*sum(zt2.*gamma.^2.*(1-(4*lambda^2)./(gamma.^2+lambda^2)));
    lambda = lambda-f/fp;
    zt1= z./((gamma.^2+lambda^2));
    zt2= zt1./((gamma.^2+lambda^2));
    f = 2*lambda*sum(zt2.*gamma.^2);
end
end
lambda = abs(lambda);