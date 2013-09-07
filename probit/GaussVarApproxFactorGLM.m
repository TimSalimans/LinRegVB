function [approxMean,approxPrec] = GaussVarApproxFactorGLM(priorMeanTimesPrec,priorPrec,features,llhfun,nrSteps)

% dimension
[n,k] = size(features);

% initialize results to zero
sumXX = zeros(3,3,n);
XXs = zeros(3,3,n);
XX = zeros(3,3,n);
for j=1:n
    XX(:,:,j) = (1e-10)*eye(3);
end
sumXY = zeros(3,n);
XY = zeros(3,n);
XYs = zeros(3,n);
natpar_meanTimesPrec = zeros(n,1);
natpar_prec = zeros(n,1);

% algorithm settings
stepSize = 1/sqrt(nrSteps);

% do stochastic approximation
for i=1:nrSteps
    
    % do regression
    if i>5
        for j=1:n
            natpar_j = XX(:,:,j)\XY(:,j);
            natpar_meanTimesPrec(j) = natpar_j(1);
            natpar_prec(j) = natpar_j(2);
        end
    end
    
    % get posterior for beta
    cholP = chol(features'*(features.*repmat(natpar_prec,1,k)) + priorPrec);
    meanTimesPrec = features'*natpar_meanTimesPrec + priorMeanTimesPrec;
    
    % sample
    betaMean = cholP\(cholP'\meanTimesPrec);
    zDraws = features*(betaMean + cholP\randn(k,1));
    
    % get log likelihood values
    llh = llhfun(zDraws);
    
    % calculate new statistics
    for j=1:n
        tj = [zDraws(j); -0.5*zDraws(j)^2; 1];
        XXs(:,:,j) = tj*tj';
        XYs(:,j) = tj*llh(j);
    end
    
    % update old statistics
    XX = (1-stepSize)*XX + stepSize*XXs;
    XY = (1-stepSize)*XY + stepSize*XYs;
    
    % do averaging if over half way
    if i > nrSteps/2
        sumXX = sumXX + XXs;
        sumXY = sumXY + XYs;
    end
end

% output final results

% do regression
for j=1:n
    natpar_j = XX(:,:,j)\XY(:,j);
    natpar_meanTimesPrec(j) = natpar_j(1);
    natpar_prec(j) = natpar_j(2);
end

% get posterior for beta
approxPrec = features'*(features.*repmat(natpar_prec,1,k)) + priorPrec;
approxMean = approxPrec\(features'*natpar_meanTimesPrec + priorMeanTimesPrec);