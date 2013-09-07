function [approxMean,approxPrec] = GaussVarApproxFactorGradientGLM(priorMeanTimesPrec,priorPrec,features,llhgrad,nrSteps)
% stochastic approximation variational Bayes for GLM's
% uses antithetics
% uses a different (cleaner) parameterization than the natural
% parameterization described in the paper

% dimension
[n,k] = size(features);

% initialize results to zero
sumXX12 = zeros(n,1);
sumXX22 = zeros(n,1);
sumXY1 = zeros(n,1);
sumXY2 = zeros(n,1);
XX12 = zeros(n,1);
XX22 = -ones(n,1);
XY1 = zeros(n,1);
XY2 = zeros(n,1);

% algorithm settings
stepSize = 1/sqrt(nrSteps);

% do stochastic approximation
for i=1:nrSteps
    
    % do regression
    natpar_meanTimesPrec = (XX22.*XY1 - XX12.*XY2)./XX22;
    natpar_prec = XY2./XX22;
    
    % get posterior for beta
    cholP = chol(features'*(features.*repmat(natpar_prec,1,k)) + priorPrec);
    meanTimesPrec = features'*natpar_meanTimesPrec + priorMeanTimesPrec;
    
    % get posterior for z=features*beta
    zMean = features*(cholP\(cholP'\meanTimesPrec));
    zVar = sum((features/cholP).^2,2);
    
    % sample - we use antithetics, this simplifies the inverses
    rn = randn(n,1);
    r = [rn -rn];
    stdv = sqrt(zVar);
    zDraws = repmat(zMean,1,2) + repmat(stdv,1,2).*r;
    
    % get gradient of log likelihood values
    llhGrad = llhgrad(zDraws);
    
    % calculate new statistics
    XXs12 = -zMean;
    XXs22 = -stdv.*rn.^2;
    XYs1 = mean(llhGrad,2);
    XYs2 = mean(r.*llhGrad,2);
    
    % update old statistics
    XX12 = (1-stepSize)*XX12 + stepSize*XXs12;
    XX22 = (1-stepSize)*XX22 + stepSize*XXs22;
    XY1 = (1-stepSize)*XY1 + stepSize*XYs1;
    XY2 = (1-stepSize)*XY2 + stepSize*XYs2;
    
    % do averaging if over half way
    if i > nrSteps/2
        sumXX12 = sumXX12 + XXs12;
        sumXX22 = sumXX22 + XXs22;
        sumXY1 = sumXY1 + XYs1;
        sumXY2 = sumXY2 + XYs2;
    end
end

% output final results
natpar_meanTimesPrec = (XX22.*XY1 - XX12.*XY2)./XX22;
natpar_prec = XY2./XX22;
approxPrec = features'*(features.*repmat(natpar_prec,1,k)) + priorPrec;
approxMean = approxPrec\(features'*natpar_meanTimesPrec + priorMeanTimesPrec);