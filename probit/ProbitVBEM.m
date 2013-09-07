function [meanVec,Prec] = ProbitVBEM(x,y,priorMeanTimesPrec,priorPrec,nrsteps) %#codegen
% Variational Bayes EM for probit regression model,
% as in Ormerod and Wand 2009

% initialize result
meanTimesPrec = priorMeanTimesPrec;
Prec = x'*x + priorPrec;

% constants
mult = sqrt(2/pi);
signy = 2*y-1;

% do updates
for i=1:nrsteps
    
    % get mean of beta
    meanbeta = Prec\meanTimesPrec;
    
    % normal ratio
    p = (x*meanbeta).*signy;
    ind = (p<-25);
    if any(any(ind))
        r = zeros(size(p));
        r(ind) = -p(ind) - 1./p(ind) + 2./p(ind).^3;
        nind = ~ind;
        r(nind) = mult*exp(-0.5*p(nind).^2)./erfc(p(nind)./-sqrt(2));
    else
        r = mult*exp(-0.5*p.^2)./erfc(p./-sqrt(2));
    end
    
    % update latents
    meanlat = x*meanbeta + signy.*r;
    
    % update beta
    meanTimesPrec = x'*meanlat + priorMeanTimesPrec;
end

% get mean
meanVec = Prec\meanTimesPrec;