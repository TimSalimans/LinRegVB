% marginal log likelihood for stochastic volatility model
function logprobs = FilterLogLike(xtau,xprec,varDraws,phiDraws) %#codegen

% dimension
nrsamples=length(varDraws);
nrvols=length(xtau);

% pre-allocate storage for Kalman filter
logprobs=zeros(nrsamples,1);

% use Kalman filter to get marginal likelihoods (vols integrated out)
for i=1:nrsamples
    
    % initialize Kalman filter
    meanM=0;
    meanX=0;
    varM=1e10; % should really be infinite, but something large will do
    varX=varDraws(i)/(1-phiDraws(i)^2);
    covMX=0;
    
    % filter
    for t=1:nrvols
        
        % prediction error & predictive variance
        v=xtau(t)/xprec(t) - meanM - meanX; 
        F=varM+varX+2*covMX+1/xprec(t);
        
        % predictive log-likelihood
        logprobs(i)=logprobs(i) - 0.5*(log(F) + v^2/F);
        
        % Kalman gain
        K1=(varM+covMX)/F;
        K2=phiDraws(i)*(varX+covMX)/F;
        
        % some intermediate quantities
        L11=1-K1;
        L12=-K1;
        L21=-K2;
        L22=phiDraws(i)-K2;
        
        % update & predict
        meanM=meanM+K1*v;
        meanX=phiDraws(i)*meanX+K2*v;
        newCovMX=varM*L21+covMX*L22;
        varM=varM*L11+covMX*L12;
        varX=phiDraws(i)*(covMX*L21+varX*L22)+varDraws(i);
        covMX=newCovMX;
    end
end
