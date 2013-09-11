% stochastic approximation VB for the stochastic volatility model
function [postParPhi,postParVar,lhVolsMeanTimesPrec,lhVolsPrec] = ApproxStochVol(ySquared,priorParPhi,priorParVar,nrSteps)

% initial posterior of model parameters is the prior
postParPhi = priorParPhi;
postParVar = [priorParVar; 0];

% base initial volatilities on characteristics of model and data
nrVols = length(ySquared);
lhVolsPrec = ones(nrVols,1)/4.93;
lhVolsMeanTimesPrec = (log(ySquared)+1.2704).*lhVolsPrec;


% initial regression statistics for the model parameters, using initial sample
[phiDraws,varDraws] = SamplePhiAndVar(postParPhi,postParVar,50);
xxPhi = diag(diag(ApproxPhiStats(phiDraws,varDraws,priorParPhi,postParPhi,priorParVar,postParVar)));
xyPhi = xxPhi*postParPhi;
xxVar = diag(diag(ApproxVarStats(phiDraws,varDraws,priorParVar,postParVar)));
xyVar = xxVar*postParVar;

% initial regression statistics for the volatilities
[avgSample,~] = ApproxVolStats(phiDraws(1),varDraws(1),lhVolsMeanTimesPrec,lhVolsPrec,ySquared);
avgGrad = lhVolsPrec-0.5;


% initialize containers for averaging
sumxyPhi = zeros(2,1);
sumxyVar = zeros(3,1);
sumxxPhi = zeros(2,2);
sumxxVar = zeros(3,3);
sumSample = zeros(nrVols,1);
sumGrad = zeros(nrVols,1);


% algorithm settings
stepSize = 1/sqrt(nrSteps);
nrOfSamples = 1;

% do stochastic approximation
for i=1:nrSteps
    
    
    % /////// sample model parameters ////////////
    [phiDraws,varDraws] = SamplePhiAndVar(postParPhi,postParVar,nrOfSamples);
    
    
    % /////// update volatilities ///////
    
    % get statistics
    [sMean,sGrad] = ApproxVolStats(phiDraws,varDraws,lhVolsMeanTimesPrec,lhVolsPrec,ySquared);
    
    % update statistics & approximation
    avgSample = (1-stepSize)*avgSample + stepSize*sMean;
    avgGrad = (1-stepSize)*avgGrad + stepSize*sGrad;
    lhVolsPrec = avgGrad + 0.5;
    lhVolsMeanTimesPrec = avgGrad + lhVolsPrec.*avgSample;
    
    
    % //////// get derivatives of approximate likelihood w.r.t. phi and var
    dif=1e-6;
    jacVar = (FilterLogLike_mex(lhVolsMeanTimesPrec,lhVolsPrec,exp(log(varDraws)+dif),phiDraws)...
        -FilterLogLike_mex(lhVolsMeanTimesPrec,lhVolsPrec,exp(log(varDraws)-dif),phiDraws))./(2*dif*varDraws);
    jacPhi = (FilterLogLike_mex(lhVolsMeanTimesPrec,lhVolsPrec,varDraws,real(tanh(atanh(phiDraws)+dif)))...
        -FilterLogLike_mex(lhVolsMeanTimesPrec,lhVolsPrec,varDraws,real(tanh(atanh(phiDraws)-dif))))./(2*dif*(1-phiDraws.^2));
        
    
    % /////// update var ///////
    
    % get statistics var
    [sxxVar,sxyVar] = ApproxVarStats(phiDraws,varDraws,priorParVar,postParVar,jacVar);
    
    % update statistics
    xxVar = (1-stepSize)*xxVar + stepSize*sxxVar;
    xyVar = (1-stepSize)*xyVar + stepSize*sxyVar;
    
    % update approximation
    postParVar = xxVar\xyVar;
    
    % /////// update phi ///////
    
    % get statistics phi
    [sxxPhi,sxyPhi] = ApproxPhiStats(phiDraws,varDraws,priorParPhi,postParPhi,priorParVar,postParVar,jacVar,jacPhi);
    
    % update statistics
    xxPhi = (1-stepSize)*xxPhi + stepSize*sxxPhi;
    xyPhi = (1-stepSize)*xyPhi + stepSize*sxyPhi;
    
    % update approximation
    postParPhi = xxPhi\xyPhi;
    
    
    % /////// do averaging if over half way /////////
    if i > nrSteps/2
        sumSample = sumSample + sMean;
        sumGrad = sumGrad + sGrad;
        sumxyPhi = sumxyPhi + sxyPhi;
        sumxxPhi = sumxxPhi + sxxPhi;
        sumxyVar = sumxyVar + sxyVar;
        sumxxVar = sumxxVar + sxxVar;
    end
end

% final update using averaged statistics
nrAvg = ceil(nrSteps/2);
lhVolsPrec = sumGrad/nrAvg + 0.5;
lhVolsMeanTimesPrec = sumGrad/nrAvg + lhVolsPrec.*sumSample/nrAvg;
postParPhi = sumxxPhi\sumxyPhi;
postParVar = sumxxVar\sumxyVar;
end

% stochastically approximate regression statistics for the volatilities
function [sMean,sGrad] = ApproxVolStats(phiDraws,varDraws,lhVolsMeanTimesPrec,lhVolsPrec,ySquared)

% do Kalman filtering and smoothing
[meanM,meanX,varM,varX,covMX] = KalmanFilterAndSmoother_mex(lhVolsMeanTimesPrec,lhVolsPrec,varDraws,phiDraws);

% calculate expectations
sumMom1 = meanM + meanX;
sumVar = varM + varX + 2*covMX;
c=exp(0.5*sumVar-sumMom1).*repmat(ySquared,1,size(sumVar,2));

% statistics
sMean = mean(sumMom1,2);
sGrad = 0.5*(mean(c,2)-1);
end

% stochastically approximate regression statistics phi
function [sxxPhi,sxyPhi] = ApproxPhiStats(phiDraws,varDraws,priorModelPhi,postParPhi,priorModelVar,postParVar,jacVar,jacPhi)

% gradient of phi w.r.t. natural parameters
betaDraws = 0.5*(phiDraws+1);
gradPhiParPhi = 2*DerivativeBetaSampler(betaDraws,postParPhi)';

% gradient of var w.r.t. natural parameters phi
gamDraws = (postParVar(2)+postParVar(3)*phiDraws.^2)./varDraws;
gradVarParPhi = gradPhiParPhi.*repmat(((2*postParVar(3)*phiDraws)./gamDraws)',2,1);

% C matrix
sxxPhi = [mean(gradPhiParPhi./repmat(phiDraws'+1,2,1),2) -mean(gradPhiParPhi./repmat(1-phiDraws',2,1),2)];

% g vector
if nargout==2 && nargin==8
    
    % derivatives of priors w.r.t. phi and var
    jacVar = jacVar - priorModelVar(1)./varDraws + priorModelVar(2)./varDraws.^2;
    jacPhi = jacPhi + priorModelPhi(1)./(phiDraws+1) - priorModelPhi(2)./(1-phiDraws);
    
    % derivatives of approximation var, w.r.t. phi and var
    jacVar = jacVar + postParVar(1)./varDraws - postParVar(2)./varDraws.^2 - postParVar(3)*(phiDraws./varDraws).^2;
    jacPhi = jacPhi + 2*postParVar(3)*phiDraws./varDraws - (2*phiDraws*postParVar(3)*(postParVar(1)+1))./(postParVar(2)+postParVar(3)*phiDraws.^2);
        
    % form g vector
    sxyPhi = mean(gradPhiParPhi.*repmat(jacPhi',2,1),2) + mean(gradVarParPhi.*repmat(jacVar',2,1),2);
end
end

% stochastically approximate regression statistics var
function [sxxVar,sxyVar] = ApproxVarStats(phiDraws,varDraws,priorModelVar,postParVar,jacVar)

% gradient of var w.r.t. its natural parameters
gamDraws = (postParVar(2)+postParVar(3)*phiDraws.^2)./varDraws;
jacGam = DerivativeGammaSampler(gamDraws,postParVar(1));
gradVarParVar = [-varDraws.*jacGam./gamDraws, 1./gamDraws, (phiDraws.^2)./gamDraws]';

% C matrix
sxxVar = [-mean(gradVarParVar./repmat(varDraws',3,1),2) ...
    mean(gradVarParVar./repmat(varDraws'.^2,3,1),2) ...
    mean(gradVarParVar.*repmat((phiDraws./varDraws)'.^2,3,1),2)];

% g vector
if nargout==2 && nargin==5
    
    % derivative of prior
    jacVar = jacVar - priorModelVar(1)./varDraws + priorModelVar(2)./varDraws.^2;
    
    % form g vector
    sxyVar = mean(gradVarParVar.*repmat(jacVar',3,1),2);
end
end

% derivative of beta sampler
% this could probably be improved
function jac = DerivativeBetaSampler(betaDraws,par)
dif = 1e-6;
logPar = log(par+1);

bpdf = exp(par(1)*log(betaDraws) + par(2)*log(1-betaDraws))./beta(par(1)+1,par(2)+1);
jac = -[(betacdf(betaDraws,exp(logPar(1)+dif),par(2)+1)-...
    betacdf(betaDraws,exp(logPar(1)-dif),par(2)+1))./(2*dif*bpdf*(par(1)+1)), ....
    (betacdf(betaDraws,par(1)+1,exp(logPar(2)+dif))-...
    betacdf(betaDraws,par(1)+1,exp(logPar(2)-dif)))./(2*dif*bpdf*(par(2)+1))];
end

% derivative of gamma sampler
% this could probably be improved
function jac = DerivativeGammaSampler(gamDraws,par)
dif = 1e-6;
logPar = log(par(1)+1);

gpdf = exp(par(1)*log(gamDraws) - gamDraws -gammaln(par(1)+1));
jac = -(gamcdf(gamDraws,exp(logPar+dif),1)-...
    gamcdf(gamDraws,exp(logPar-dif),1))./(2*dif*gpdf*(par(1)+1));
end
