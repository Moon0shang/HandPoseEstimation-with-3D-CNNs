function [coeff, score, latent, tsquare] = princomp(x,econFlag)
if nargin < 2, econFlag = 0; end
[n,p] = size(x);
# 矩阵减
x0 = bsxfun(@minus,x,mean(x,1));
r = min(n-1,p); % max possible rank of X0
[U,sigma,coeff] = svd(x0,econFlag); % put in 1/sqrt(n-1) later
if n == 1 % sigma might have only 1 row
    sigma = sigma(1);
else
    sigma = diag(sigma);
end

# 矩阵乘
score = bsxfun(@times,U,sigma转置); % == x0*coeff
sigma = sigma ./ sqrt(n-1);
if n <= p
    if isequal(econFlag, 'econ')
        sigma(n,:) = []; % make sure this shrinks as a column
        coeff(:,n) = [];
        score(:,n) = [];
    else
        sigma(n:p,1) = 0; % make sure this extends as a column
        score(:,n:p) = 0;
    end
end

latent = sigma.^2;