function [lambda,conv] = ChiSqx0_GSVD2(U1,U2,r,ups,M,za,p)
% Use the central Chi^2 test to select lambda using the GSVD
%
% Inputs:
% U1, U2: matrices from the GSVD of {A1,L1} and {A2,L2}
% r: Value of r in the chi^2 test (b-A*x0)
% ups, M: vectors of the upsilon and mu values from the GSVD
% za: Critical value of z_(1-alpha/2)
% p: Number of rows in L
%
% Outputs:
% lambda: Regularization parameter selected by chi^2 test
% conv: indicator of whether the method converged with Newton

m = length(ups);
n = m;
lambda = 1;
gamma = ups(1:n)./M;

s = U2'*reshape(r,sqrt(n),sqrt(n))*U1;
s = s(:);
dof = m-n + min(n,p);
mt = dof - norm(s(n+1:m))^2;
s = s(1:n);
st= s./(gamma.^2+lambda^2);

f = lambda^2*(s'*st)-mt;
iter = 0; 
while abs(f) > sqrt(2*(dof))*za && iter < 100
fp = 2*lambda*norm(st.*gamma,2)^2;
lambda = lambda-f/fp;
st= s./(gamma.^2+lambda^2);
f = lambda^2*(s'*st) - mt;
iter = iter + 1;
end
conv = 1;

if iter ==50 || abs(lambda) >1e10
    conv = 0;
    lambda = 1;
s2 = s.^2;
zt1= s2./((gamma.^2+lambda^2));
zt2= zt1./((gamma.^2+lambda^2));
f = 2*lambda*sum(zt2.*gamma.^2);
    iter = 0; 
while abs(f) > sqrt(2*(dof))*za && iter < 50
    fp = 2*sum(zt2.*gamma.^2.*(1-(4*lambda^2)./(gamma.^2+lambda^2)));
    lambda = lambda-f/fp;
    zt1= s2./((gamma.^2+lambda^2));
    zt2= zt1./((gamma.^2+lambda^2));
    f = 2*lambda*sum(zt2.*gamma.^2);
end
end
lambda = abs(lambda);