function [W0,W1,W2] = Framelet02(nn)
% Forms the 1D framelet matrix such that [W0;W1;W2] has orthonormal columns
% W0 is a low-pass filter while W1 and W2 are high-pass filters.

    W0 = diag(2*ones(nn,1),0) + diag(ones(nn-1,1),-1) + diag(ones(nn-1,1),1);
    W0(1,1) = 3; W0(end,end) = 3;
    W0 = 0.25*W0;

    W1 = diag(-ones(nn-1,1),-1) + diag(ones(nn-1,1),1);
    W1(1,1) = -1; W1(end,end) = 1;
    W1 = (sqrt(2)/4)*W1;

    W2 = diag(2*ones(nn,1),0) + diag(-ones(nn-1,1),-1) + diag(-ones(nn-1,1),1);
    W2(1,1) = 1; W2(end,end) = 1;
    W2 = 0.25*W2;
end
