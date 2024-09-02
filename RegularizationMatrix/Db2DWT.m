function [W1T,W2T] = Db2DWT(nn)
% Forms the 1D D4 (Db2) wavelet transform matrices W1T and W2T for a given
% number of points nn.

    h0 = 1+sqrt(3);
    h1 = 3+sqrt(3);
    h2 = 3-sqrt(3);
    h3 = 1-sqrt(3);
    S = [h0, h1, h2, h3]; T = [h3, -h2, h1, -h0];
    R = zeros(nn,nn);
    if nn/2-floor(nn/2) == 0
        for j=1:floor(nn/2)-1
            R(j,2*j-1:2*j+2) = S;
        end
        R(floor(nn/2),2*floor(nn/2)-1:2*floor(nn/2)) = S(1:2);
        R(floor(nn/2),1:2) = S(3:4);
        for j=floor(nn/2)+1:nn-1
             R(j,2*(j-floor(nn/2))-1:2*(j-floor(nn/2))+2) = T;
        end
    else
        for j=1:floor(nn/2)-1
            R(j,2*j-1:2*j+2) = S;
        end
        R(floor(nn/2),2*floor(nn/2)-1:2*floor(nn/2)) = S(1:2);
        R(floor(nn/2),1:2) = S(3:4);
        for j=floor(nn/2)+1:nn-2
            R(j,2*(j-floor(nn/2))-1:2*(j-floor(nn/2))+2) = T;
        end
    end
    R(nn,end-1:end) = T(1:2);
    R(nn,1:2) = T(3:4);
    R = R/(4*sqrt(2));
    W1T = R(1:nn/2,:);
    W2T = R(nn/2+1:end,:);
    W1T = sparse(W1T);
    W2T = sparse(W2T);
end