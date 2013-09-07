function [approxMean,approxPrec] = GaussVarApproxBasic(guessMean,guessPrec,logpostfun,nrSteps)

% dimension
k = length(guessMean);
nvp = k + k*(k+1)/2;

% initialize results to zero
sumXX = zeros(nvp+1,nvp+1);
sumXY = zeros(nvp+1,1);

% determine indices for conversions
[rind,cind] = find(triu(ones(k,k)-eye(k)));
nondiagnr = sub2ind([k k],rind,cind);
indexNatparToMeanTimesPrec = 1:k;
indexNatparToPrecisionMatrix = (k+1):nvp;
indexPrecisionMatrixToNatpar = [sub2ind([k k],1:k,1:k)'; nondiagnr];

% initial statistics for stochastic approximation
natpar = [guessPrec*guessMean; diag(guessPrec); guessPrec(nondiagnr); 0];
triuP = triu(guessPrec);
XX = (1e-10)*eye(nvp+1);
XY = (1e-10)*natpar;

% algorithm settings
stepSize = 1/sqrt(nrSteps);

% do stochastic approximation
for i=1:nrSteps
    
    % do regression
    if i >= 3*nvp
       natpar = XX\XY;
    end
    
    % cholesky of current inverse variance matrix
    triuP(indexPrecisionMatrixToNatpar) = natpar(indexNatparToPrecisionMatrix);
    cholP = chol(triuP);
    
    % meanTimesPrec
    meanTimesPrec = natpar(indexNatparToMeanTimesPrec);
    
    % sample
    xDraws = cholP\(cholP'\meanTimesPrec) + cholP\randn(k,1);
    
    % get log posterior values
    logPost = logpostfun(xDraws);    
    
    % calculate new statistics
    xDraws = xDraws';
    xmat = [xDraws -0.5*xDraws.^2 -xDraws(:,rind).*xDraws(:,cind) 1];
    XXs = xmat'*xmat;
    XYs = (logPost*xmat)';
    
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
natpar = sumXX\sumXY;
triuP(indexPrecisionMatrixToNatpar) = natpar(indexNatparToPrecisionMatrix);
approxPrec = triuP+triuP'-diag(diag(triuP));
approxMean = approxPrec\natpar(indexNatparToMeanTimesPrec);