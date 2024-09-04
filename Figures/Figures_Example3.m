% Figures 7-9: Code that solves Example 3 using SB and MM with framelets 
% or L_D for regularization.

clear, close all
set(0,'DefaultAxesFontSize', 16)
set(0, 'DefaultLineLineWidth', 1.5)
set(0, 'DefaultLineMarkerSize', 4)
%% Setup the Problem
n=128; 
rng(10)

% Matrix A
band=15; sigma1 = 1.5;
Ab = exp(-([0:band-1].^2)/(2*sigma1^2));
z = [Ab, zeros(1,n-2*band+1), fliplr(Ab(2:end))];
A1 = (1/(sqrt(2*pi)*sigma1))*toeplitz(z,z);

band=15; sigma2 = 0.8;
Ab = exp(-([0:band-1].^2)/(2*sigma2^2));
z = [Ab, zeros(1,n-2*band+1), fliplr(Ab(2:end))];
A2 = (1/(sqrt(2*pi)*sigma2))*toeplitz(z,z);
m = size(A1,1)*size(A2,1);
lang=n;
x_true = imread('barcode.gif');
x_true = double(x_true);
x_true = [251*ones(16,128) ;251*ones(95,6) x_true(1:95,1:116) 251*ones(95,6);251*ones(17,128)];
x_true = x_true./max(max(x_true)); 
X_true = x_true;
x_true = x_true(:);

% Add noise to Ax = b
Ax      = A2*(X_true)*A1';
Ax = Ax(:);
b_true = Ax;
err_lev = 10; 
sigma   = err_lev/100 * norm(Ax) / sqrt(m);
eta     =  sigma * randn(m,1);
b       = Ax + eta; % Blurred signal: b + E
SNR = 20*log10(norm(Ax)/norm(b-Ax));

BT = reshape(b_true,[],lang);
B = reshape(b,[],lang);
XT = reshape(x_true,[],lang);

%% Plot PSF
np=9; 
band=4; sigma1 = 1.5;
Ab = exp(-([0:band-1].^2)/(2*sigma1^2));
z = [Ab, zeros(1,np-band)];
A1p = (1/(sqrt(2*pi)*sigma1))*toeplitz(z,z);

band=4; sigma2 = 0.8;
Ab = exp(-([0:band-1].^2)/(2*sigma2^2));
z = [Ab, zeros(1,np-band)];
A2p = (1/(sqrt(2*pi)*sigma2))*toeplitz(z,z);

xp = zeros(np);
xp(floor(np/2)+1,floor(np/2)+1)=1;
Ap = A2p*xp*A1p';

%% Figure 7: Plot x_true, PSF, b_true, and b
% Figure 7(a)
figure(1), imshow(XT, [], 'initialmagnification', 100000, 'border', 'tight')
% Figure 7(b)
figure(2), imshow(Ap, [], 'initialmagnification', 100000, 'border', 'tight')
% Figure 7(c)
figure(3), imshow(BT, [], 'initialmagnification', 100000, 'border', 'tight')
% Figure 7(d)
figure(4), imshow(B, [], 'initialmagnification', 100000, 'border', 'tight')

% Rescale
N = ones(n-1,1);
A1 = A1*(1/sqrt(sigma));
A2 = A2*(1/sqrt(sigma));
b = b*(1/sigma);

[W0,W1,W2] = Framelet02(n);
L = [W0;W1;W2];
p = size(L,1);

[U1,SA1,VA1] = svd(A1);
U1 = fliplr(U1);
SA1 = flipud(diag(SA1));
VA1 = fliplr(VA1);
UpsD1 = SA1./sqrt(1+SA1.^2);
MD1 = ones(length(SA1),1)./sqrt(1+SA1.^2);
V1 = L*VA1;
X1i = diag(sqrt(1+SA1.^2))*VA1';
X1 = VA1*diag(MD1);
[U2,SA2,VA2] = svd(A2);
U2 = fliplr(U2);
SA2 = flipud(diag(SA2));
VA2 = fliplr(VA2);
UpsD2 = SA2./sqrt(1+SA2.^2);
MD2 = ones(length(SA2),1)./sqrt(1+SA2.^2);
V2 = L*VA2;
X2i = diag(sqrt(1+SA2.^2))*VA2';
X2 = VA2*diag(MD2);

UpsK = kron(UpsD1,UpsD2);
MK = kron(MD1,MD2);

%% Find the optimal lambda for SB
tau = 0.02;
tol = 0.01; 
lambdavec2 = logspace(-1,2,91)';
XSBo = ones(length(lambdavec2),40);
XSB = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2)
    [~,XSB1,~,~] = SBM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,lambdavec2(i)*ones(40,1),tau,tol,40);
    for j=1:size(XSB1,2)
    XSBo(i,j) = norm(XSB1(:,j)-x_true)/norm(x_true);
    end
    XSB(i,1) = XSBo(i,j);
end
[~,iSB] = min(XSB);
LSBmF = lambdavec2(iSB);

% Run SB with the parameters selected every iteration
iter = 20;
za = 0.0013;
[xGF,XGF,~,~,LGF,~] = SBM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'gcv',tau,tol,0,iter,za);
[xCCF,XCCF,~,~,LCCF,~] = SBM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'cchi',tau,tol,0,iter,za);
[xCF,XCF,~,~,LCF,~] = SBM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'ncchi',tau,tol,0,iter,za);
[xCLF,XCLF,DOF,GOF] = SBM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,LSBmF,tau,tol,iter);

% SB Framelet: RE and ISNR
RREGF = zeros(size(XGF,2),1);
RRECF = zeros(size(XCF,2),1);
RRECCF = zeros(size(XCCF,2),1);
RRECLF = zeros(size(XCLF,2),1);
ISGF = zeros(size(XGF,2),1);
ISCF = zeros(size(XCF,2),1);
ISCCF = zeros(size(XCCF,2),1);
ISCLF = zeros(size(XCLF,2),1);
Inum = norm(b-x_true,2);

for j=1:size(XGF,2)
    RREGF(j,1) = norm(XGF(:,j) - x_true)/norm(x_true);
    ISGF(j,1) = 20*log10(Inum/norm(XGF(:,j)-x_true));
end
for j=1:size(XCF,2)
    RRECF(j,1) = norm(XCF(:,j) - x_true)/norm(x_true);
    ISCF(j,1) = 20*log10(Inum/norm(XCF(:,j)-x_true));
end
for j=1:size(XCCF,2)
    RRECCF(j,1) = norm(XCCF(:,j) - x_true)/norm(x_true);
    ISCCF(j,1) = 20*log10(Inum/norm(XCCF(:,j)-x_true));
end
for j=1:size(XCLF,2)
    RRECLF(j,1) = norm(XCLF(:,j) - x_true)/norm(x_true);
    ISCLF(j,1) = 20*log10(Inum/norm(XCLF(:,j)-x_true));
end

% Find the optimal lambda for MM
ep = 0.02; 
lambdavec2 = logspace(-1,2,91)';
XMMo = ones(length(lambdavec2),40);
xMM = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2)
    [~,XMM1] = MM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,lambdavec2(i)*ones(40,1),ep,tol,40);
    for j=1:size(XMM1,2)
    XMMo(i,j) = norm(XMM1(:,j)-x_true)/norm(x_true);
    end
    xMM(i,1) = XMMo(i,j);
end
[~,iMM] = min(xMM);
LMMmF = lambdavec2(iMM);

% Run MM with the parameters selected every iteration
iter = 20; 
tol = 0.01;
za = 0.0013;
[x2GF,X2GF,L2GF,~] = MM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'gcv',ep,tol,0,iter,za);
[x2CCF,X2CCF,L2CCF,~] = MM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'cchi',ep,tol,0,iter,za);
[x2CF,X2CF,L2CF,~] = MM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'ncchi',ep,tol,0,iter,za);
[x2CLF,X2CLF,HOF] = MM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,LMMmF,ep,tol,iter);

% MM Framelets: RE and ISNR
RREG2F = zeros(size(X2GF,2),1);
RREC2F = zeros(size(X2CF,2),1);
RRECC2F = zeros(size(X2CCF,2),1);
RRECL2F = zeros(size(X2CLF,2),1);
ISG2F = zeros(size(X2GF,2),1);
ISC2F = zeros(size(X2CF,2),1);
ISCC2F = zeros(size(X2CCF,2),1);
ISCL2F = zeros(size(X2CLF,2),1);

for j=1:size(X2GF,2)
    RREG2F(j,1) = norm(X2GF(:,j) - x_true)/norm(x_true);
    ISG2F(j,1) = 20*log10(Inum/norm(X2GF(:,j)-x_true));
end
for j=1:size(X2CF,2)
    RREC2F(j,1) = norm(X2CF(:,j) - x_true)/norm(x_true);
    ISC2F(j,1) = 20*log10(Inum/norm(X2CF(:,j)-x_true));
end
for j=1:size(X2CCF,2)
    RRECC2F(j,1) = norm(X2CCF(:,j) - x_true)/norm(x_true);
    ISCC2F(j,1) = 20*log10(Inum/norm(X2CCF(:,j)-x_true));
end
for j=1:size(X2CLF,2)
    RRECL2F(j,1) = norm(X2CLF(:,j) - x_true)/norm(x_true);
    ISCL2F(j,1) = 20*log10(Inum/norm(X2CLF(:,j)-x_true));
end

%% L_D Regularization
La = -diag(ones(n,1)./sqrt(2),0) + diag(ones(n-1,1)./sqrt(2),1); La(end,1) = 1/sqrt(2);

[U1,V1,X11,Ups1,M1] = gsvd(A1,La);
X1i = X11';
X1 = (eye(size(X11))/X11)';
UpsD1 = diag(Ups1);
MD1 = diag(M1);
[U2,V2,X22,Ups2,M2] = gsvd(A2,eye(n));
X2i = X22';
X2 = (eye(size(X22))/X22)';
UpsD2 = diag(Ups2);
MD2 = diag(M2);

UpsK = kron(UpsD1,UpsD2);
MK = kron(MD1,MD2);

% Find the optimal lambda for SB
lambdavec2 = logspace(-1,2,91)';
XSBo = ones(length(lambdavec2),40);
XSB = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2)
    [~,XSB1,~,~] = SBM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,lambdavec2(i)*ones(40,1),tau,tol,40);
    for j=1:size(XSB1,2)
    XSBo(i,j) = norm(XSB1(:,j)-x_true)/norm(x_true);
    end
    XSB(i,1) = XSBo(i,j);
end
[~,iSB] = min(XSB);
LSBm = lambdavec2(iSB);

% Run SB with the parameters selected every iteration
iter = 20;
za = 0.0013;
[xG,XG,~,~,LG,~] = SBM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'gcv',tau,tol,0,iter,za);
[xCC,XCC,~,~,LCC,~] = SBM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'cchi',tau,tol,0,iter,za);
[xC,XC,~,~,LC,~] = SBM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'ncchi',tau,tol,0,iter,za);
[xCL,XCL,DO,GO] = SBM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,LSBm,tau,tol,iter);

% SB Wavelet: RE and ISNR
RREG = zeros(size(XG,2),1);
RREC = zeros(size(XC,2),1);
RRECC = zeros(size(XCC,2),1);
RRECL = zeros(size(XCL,2),1);
ISG = zeros(size(XG,2),1);
ISC = zeros(size(XC,2),1);
ISCC = zeros(size(XCC,2),1);
ISCL = zeros(size(XCL,2),1);
Inum = norm(b-x_true,2);

for j=1:size(XG,2)
    RREG(j,1) = norm(XG(:,j) - x_true)/norm(x_true);
    ISG(j,1) = 20*log10(Inum/norm(XG(:,j)-x_true));
end
for j=1:size(XC,2)
    RREC(j,1) = norm(XC(:,j) - x_true)/norm(x_true);
    ISC(j,1) = 20*log10(Inum/norm(XC(:,j)-x_true));
end
for j=1:size(XCC,2)
    RRECC(j,1) = norm(XCC(:,j) - x_true)/norm(x_true);
    ISCC(j,1) = 20*log10(Inum/norm(XCC(:,j)-x_true));
end
for j=1:size(XCL,2)
    RRECL(j,1) = norm(XCL(:,j) - x_true)/norm(x_true);
    ISCL(j,1) = 20*log10(Inum/norm(XCL(:,j)-x_true));
end

% Find the optimal lambda for MM
lambdavec2 = logspace(-1,2,91)';
XMMo = ones(length(lambdavec2),40);
xMM = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2)
    [~,XMM1] = MM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,lambdavec2(i)*ones(40,1),ep,tol,40);
    for j=1:size(XMM1,2)
    XMMo(i,j) = norm(XMM1(:,j)-x_true)/norm(x_true);
    end
    xMM(i,1) = XMMo(i,j);
end
[~,iMM] = min(xMM);
LMMm = lambdavec2(iMM);

% Run MM with the parameters selected every iteration
iter = 20; 
tol = 0.01;
za = 0.0013;
[x2G,X2G,L2G,~] = MM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'gcv',ep,tol,0,iter,za);
[x2CC,X2CC,L2CC,~] = MM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'cchi',ep,tol,0,iter,za);
[x2C,X2C,L2C,~] = MM_ParamSel_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,'ncchi',ep,tol,0,iter,za);
[x2CL,X2CL,HO] = MM_GSVD2(U1,U2,V1,V2,X1,X2,X1i,X2i,UpsK,MK,b,LMMm,ep,tol,iter);

% MM Wavelet: RE and ISNR
RREG2 = zeros(size(X2G,2),1);
RREC2 = zeros(size(X2C,2),1);
RRECC2 = zeros(size(X2CC,2),1);
RRECL2 = zeros(size(X2CL,2),1);
ISG2 = zeros(size(X2G,2),1);
ISC2 = zeros(size(X2C,2),1);
ISCC2 = zeros(size(X2CC,2),1);
ISCL2 = zeros(size(X2CL,2),1);

for j=1:size(X2G,2)
    RREG2(j,1) = norm(X2G(:,j) - x_true)/norm(x_true);
    ISG2(j,1) = 20*log10(Inum/norm(X2G(:,j)-x_true));
end
for j=1:size(X2C,2)
    RREC2(j,1) = norm(X2C(:,j) - x_true)/norm(x_true);
    ISC2(j,1) = 20*log10(Inum/norm(X2C(:,j)-x_true));
end
for j=1:size(X2CC,2)
    RRECC2(j,1) = norm(X2CC(:,j) - x_true)/norm(x_true);
    ISCC2(j,1) = 20*log10(Inum/norm(X2CC(:,j)-x_true));
end
for j=1:size(X2CL,2)
    RRECL2(j,1) = norm(X2CL(:,j) - x_true)/norm(x_true);
    ISCL2(j,1) = 20*log10(Inum/norm(X2CL(:,j)-x_true));
end

%% Figure 8: Wavelet/Framelet Solutions

% Fig. 8(a): SB, Framelet, Optimal
xx = reshape(XCLF(:,end),n,n);
figure(5), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(b): SB, Wavelet, Optimal
xx = reshape(XCL(:,end),n,n);
figure(6), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(c): MM, Framelet, Optimal
xx = reshape(X2CLF(:,end),n,n);
figure(7), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(d): MM, Wavelet, Optimal
xx = reshape(X2CL(:,end),n,n);
figure(8), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(e): SB, Framelet, GCV
xx = reshape(XGF(:,end),n,n);
figure(9), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(f): SB, Wavelet, GCV
xx = reshape(XG(:,end),n,n);
figure(10), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(g): MM, Framelet, GCV
xx = reshape(X2GF(:,end),n,n);
figure(11), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(h): MM, Wavelet, GCV
xx = reshape(X2G(:,end),n,n);
figure(12), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(i): SB, Framelet, chi^2
xx = reshape(XCF(:,end),n,n);
figure(13), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(j): SB, Wavelet, chi^2
xx = reshape(XC(:,end),n,n);
figure(14), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(k): MM, Framelet, chi^2
xx = reshape(X2CF(:,end),n,n);
figure(15), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

% Fig. 8(l): MM, Wavelet, chi^2
xx = reshape(X2C(:,end),n,n);
figure(16), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

%% Figure 9: RE and ISNR plots

% Figure 9(a)
figure(17), plot(1:size(XCLF,2),RRECLF,'-^',1:size(XGF,2),RREGF,'-|',1:size(XCCF,2),RRECCF,'-o',1:size(XCF,2),RRECF,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 0.2 0.45])

% Figure 9(b)
figure(18), plot(1:size(XCL,2),RRECL,'-^',1:size(XG,2),RREG,'-|',1:size(XCC,2),RRECC,'-o',1:size(XC,2),RREC,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 0.2 0.45])

% Figure 9(c)
figure(19), plot(1:size(X2CLF,2),RRECL2F,'-^',1:size(X2GF,2),RREG2F,'-|',1:size(X2CCF,2),RRECC2F,'-o',1:size(X2CF,2),RREC2F,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 0.2 0.45])

% Figure 9(d)
figure(20), plot(1:size(X2CL,2),RRECL2,'-^',1:size(X2G,2),RREG2,'-|',1:size(X2CC,2),RRECC2,'-o',1:size(X2C,2),RREC2,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 0.2 0.45])

% Figure 9(e)
figure(21), plot(1:size(XCLF,2),ISCLF,'-^',1:size(XGF,2),ISGF,'-|',1:size(XCCF,2),ISCCF,'-o',1:size(XCF,2),ISCF,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 28 36])

% Figure 9(f)
figure(22), plot(1:size(XCL,2),ISCL,'-^',1:size(XG,2),ISG,'-|',1:size(XCC,2),ISCC,'-o',1:size(XC,2),ISC,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 28 36])

% Figure 9(g)
figure(23), plot(1:size(X2CLF,2),ISCL2F,'-^',1:size(X2GF,2),ISG2F,'-|',1:size(X2CCF,2),ISCC2F,'-o',1:size(X2CF,2),ISC2F,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 28 36])

% Figure 9(h)
figure(24), plot(1:size(X2CL,2),ISCL2,'-^',1:size(X2G,2),ISG2,'-|',1:size(X2CC,2),ISCC2,'-o',1:size(X2C,2),ISC2,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])
axis([0 20 28 36])