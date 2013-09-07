function [approxMean,approxPrec] = GaussVarApproxHessianMinibatch(guessMean,guessPrec,gradFun,nrSteps)

% dimension
k=length(guessMean);

% initialize results to zero
sumSample = zeros(k,1);
sumGrad = zeros(k,1);
sumHess = zeros(k,k);

% initial statistics for stochastic approximation
P = guessPrec;
z = guessMean;
a = zeros(k,1);

% algorithm settings
stepSize = 1/sqrt(nrSteps*10);

% do stochastic approximation
for i=1:nrSteps
    for j=randperm(10)
        
        % cholesky of current inverse variance matrix
        cholP = chol(P);
        
        % sample
        xMean = cholP\(cholP'\a) + z;
        betaDraws = xMean + cholP\randn(k,1);
        
        % get average gradient and Hessian
        [logPostGrad,logPostHess] = gradFun(betaDraws,j);
        
        % update statistics
        P = (1-stepSize)*P - stepSize*logPostHess;
        a = (1-stepSize)*a + stepSize*logPostGrad;
        z = (1-stepSize)*z + stepSize*betaDraws;
        
        % do averaging if over half way
        if i > nrSteps/2
            sumSample = sumSample + betaDraws;
            sumGrad = sumGrad + logPostGrad;
            sumHess = sumHess + logPostHess;
        end
    end
end

% output final results
approxMean = sumSample/ceil(nrSteps/2)/10 - sumHess\sumGrad;
approxPrec = -sumHess/ceil(nrSteps/2)/10;