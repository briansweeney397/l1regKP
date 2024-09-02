function [reg_min,G,reg_param] = gcvIterGSVD2(U1,U2,V1,V2,s,b,h)
% Adapted from gcv in regularization tools by
% Per Christian Hansen which has the following license:
%
% Copyright (c) 2015, Per Christian Hansen
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution
% * Neither the name of DTU Compute nor the names of its
%   contributors may be used to endorse or promote products derived from this
%   software without specific prior written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%
% Use GCV function to find optimal lambda for the problem
% ||Ax-b||_2^2 + lambda^2||Lx-h||_2^2,
% where A = kron(A1,A2) and L = kron(L1,L2) have KP structure.
% 
% Inputs:
% U1 and V1 are from the GSVD of {A1,L1}
% U2 and V2 are from the GSVD of {A2,L2}
% s is from the KP of the upsilon and mu in the GSVDs, with s = [sigma,mu]
% b and h are the right-hand sides
%
% If any output arguments are specified, then the minimum of G is
% identified and the corresponding reg. parameter reg_min is returned.

% Set defaults
npoints = 200;                      % Number of points on the curve.
smin_ratio = 16*eps;                % Smallest regularization parameter.

% Initialization.
m = size(U1,1)*size(U2,1); p1= size(V1,1)^2; n=m; [~,ps] = size(s);
beta = reshape(U2'*reshape(b,sqrt(n),sqrt(n))*U1,n,1); beta2 = norm(b)^2 - norm(beta)^2;
hhat = V2'*reshape(h,sqrt(p1),sqrt(p1))*V1;
hhat = hhat(1:sqrt(n),1:sqrt(n));
hhat= hhat(:);
ind = find(s(:,2)>eps);
beta = beta(ind);
hhat = hhat(ind);
s = s(ind,:);
if (ps==2)
  s = s(length(ind):-1:1,1)./s(length(ind):-1:1,2); beta = beta(length(ind):-1:1); hhat = hhat(length(ind):-1:1);
end

  % Vector of regularization parameters.
  reg_param = zeros(npoints,1); G = reg_param; s2 = s.^2;
    reg_param(npoints) = max([s(length(ind)),s(1)*smin_ratio]);
    ratio = (s(1)/reg_param(npoints))^(1/(npoints-1));
  for i=npoints-1:-1:1, reg_param(i) = ratio*reg_param(i+1); end

  % Intrinsic residual.
  delta0 = 0;
  if (m > n & beta2 > 0), delta0 = beta2; end

  % Vector of GCV-function values.
  for i=1:npoints
    G(i) = gcvfunIter(reg_param(i),s2,beta(1:length(ind)), hhat,delta0,m-n);
  end 

  % Find minimum, if requested.
    [~,minGi] = min(G); % Initial guess.
    reg_min = fminbnd('gcvfunIter',...
      reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),...
      optimset('Display','off'),s2,beta(1:length(ind)),hhat,delta0,m-n); % Minimizer.
end