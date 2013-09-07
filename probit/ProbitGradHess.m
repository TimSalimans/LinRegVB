function [grad,hess] = ProbitGradHess(signedx,priorMeanTimePrec,priorPrec,beta,nr_of_minibatch) %#codegen
% input 'beta' is k times p matrix containing p samples of k parameters
% return average gradient and hessian of log-posterior for probit regression model
nrSamples=size(beta,2);

% subsampling
if nargin == 5
    n = size(signedx,1);
    selected = round((nr_of_minibatch-1)*n/10+1):round(nr_of_minibatch*n/10);
    submult = n/length(selected);
    signedx = signedx(selected,:);
else
    submult = 1;
end

% gradient of log-likelihood
grad = priorMeanTimePrec - priorPrec*mean(beta,2);
mult=sqrt(2/pi);
p = signedx*beta;
ind = (p<-25);
hasoutliers = any(any(ind));
if hasoutliers
    nind = ~ind;
    r = zeros(size(p));
    r(ind) = -p(ind) - 1./p(ind) + 2./p(ind).^3;
    r(nind) = mult*exp(-0.5*p(nind).^2)./erfc(p(nind)./-sqrt(2));
else
    r = mult*exp(-0.5*p.^2)./erfc(p./-sqrt(2));
end
grad = grad + submult*signedx'*sum(r,2)/nrSamples;

% hessian of log-likelihood
if nargout==2
    hess = -priorPrec;
    if hasoutliers
        d = zeros(size(p));
        d(ind) = -1 + 1./p(ind).^2 - 6./p(ind).^4;
        d(nind) = -p(nind).*r(nind) - r(nind).^2;
    else
        d = -p.*r - r.^2;
    end
    hess = hess + submult*signedx'*(signedx.*repmat(sum(d,2)/nrSamples,1,size(signedx,2)));
end