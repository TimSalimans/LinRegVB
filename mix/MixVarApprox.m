function [mixWeights,mixMeans,mixPrecs] = MixVarApprox(logPostFun,gradfun,nrComponents,nrSteps,x0)

% algorithm settings
stepSize = 1/sqrt(nrSteps);
nrSamples = nrComponents;

% dimension
k = length(x0);

% do optimization to find the mode
opt = optimset('Display','off','LargeScale','off');
[xMode,~,~,~,~,xNegHess] = fminunc(@(x)(-logPostFun(x)),x0,opt);

% use maximum + curvature for initialization
cholNegHess = chol(xNegHess);
mixMeans = cell(nrComponents,1);
mixPrecs = cell(nrComponents,1);
for c=1:nrComponents
    mixMeans{c} = xMode + (cholNegHess\randn(k,1));
    mixPrecs{c} = xNegHess*nrComponents/3;
end
mixWeights = ones(1,nrComponents)/nrComponents;

% pre-allocate regression statistics
xxWeights = ones(nrComponents,1);
xyWeights = log(mixWeights)';
C = ones(nrComponents,1);
H = cell(nrComponents,1);
z = cell(nrComponents,1);
a = cell(nrComponents,1);
for c=1:nrComponents
    H{c} = -mixPrecs{c};
    z{c} = mixMeans{c};
    a{c} = zeros(k,1);
end

% do stochastic approximation
for i=1:nrSteps
    
    % ///////// get samples from mixture //////////
    [xSampled,sampDensPerComp] = SampleMixture(mixWeights,mixMeans,mixPrecs,nrSamples);
    
    
    % /////// get total density + Rao-Blackwellized component indicator ///
    [RBindicator,totalSampDens] = CombineMixtureComponents(mixWeights,sampDensPerComp);
    
    
    % //////// get exact log posterior density + gradient + Hessian ///////
    lpDens = zeros(nrSamples,1);
    grad = cell(nrSamples,1);
    hess = cell(nrSamples,1);
    for j=1:nrSamples
        lpDens(j) = logPostFun(xSampled(:,j));
        [grad{j},hess{j}] = gradfun(xSampled(:,j));
    end
    
    
    % ////////// update mixture weights /////////////
    
    % xx matrix
    sxxWeights = mean(RBindicator,2);
    
    % log q(z)
    sxyWeights = sxxWeights.*log(mixWeights)';
    
    % log p(x) - log q(x)
    sxyWeights = sxyWeights + RBindicator*(lpDens-log(totalSampDens)')/nrSamples;
    
    % update statistics
    xxWeights = (1-stepSize)*xxWeights + stepSize*sxxWeights;
    xyWeights = (1-stepSize)*xyWeights + stepSize*sxyWeights;
    
    % update variational parameters
    logWeights = (xyWeights./xxWeights)';
    mixWeights = exp(logWeights-max(logWeights));
    mixWeights = mixWeights/sum(mixWeights);
    
    
    % /////// update the component indicator //////////
    [RBindicator,~] = CombineMixtureComponents(mixWeights,sampDensPerComp);
    
    
    % //////// update parameters of mixture components ///////////
    
    % loop over components
    for c=1:nrComponents
        
        % sampled statistics for this component
        sc = sum(RBindicator(c,:));
        sz = xSampled*RBindicator(c,:)';
        sH = zeros(k,k);
        sa = zeros(k,1);
        for j=1:nrSamples
            
            % derivatives of log RB indicator
            [grb,hrb] = DerivativesLogRBind(xSampled(:,j),RBindicator(:,j),mixMeans,mixPrecs,c);
            
            % sH and sa stats
            sa = sa + RBindicator(c,j)*(grad{j}+grb);
            sH = sH + RBindicator(c,j)*(hess{j}+hrb);
        end
        
        % update running stats
        C(c) = (1-stepSize)*C(c) + stepSize*sc;
        H{c} = (1-stepSize)*H{c} + stepSize*sH;
        z{c} = (1-stepSize)*z{c} + stepSize*sz;
        a{c} = (1-stepSize)*a{c} + stepSize*sa;
        
        % update variational parameters
        mixPrecs{c} = -H{c}/C(c);
        mixMeans{c} = -H{c}\a{c} + z{c}/C(c);
        
        % update densities for this component
        cp = chol(mixPrecs{c});
        sampDensPerComp(c,:) = exp(-(k/2)*log(2*pi)+sum(log(diag(cp))) ...
            -0.5*sum((cp*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
        
        % update the component indicator
        [RBindicator,~] = CombineMixtureComponents(mixWeights,sampDensPerComp);
    end
end
end

% sample from the approximate posterior
function [xSampled,sampDensPerComp] = SampleMixture(mixWeights,mixMeans,mixPrecs,nrSamples)

% dimensions
k = length(mixMeans{1});
nrComponents = length(mixMeans);

% get cholesky's of the precision matrices of all Gaussian components
cholPrec = cell(nrComponents,1);
for c=1:nrComponents
    cholPrec{c} = chol(mixPrecs{c});
end

% sample mixture component indicators
cumsumProbs = cumsum(mixWeights);
comps = 1+sum(repmat(rand(nrSamples,1),1,nrComponents)>repmat(cumsumProbs,nrSamples,1),2);

% sample from the Gaussian mixture components
xSampled = zeros(k,nrSamples);
z = randn(k,nrSamples);
for j=1:nrSamples
    xSampled(:,j) = mixMeans{comps(j)} + cholPrec{comps(j)}\z(:,j);
end

% get densities at the sampled points
sampDensPerComp = zeros(nrComponents,nrSamples);
for c=1:nrComponents
    sampDensPerComp(c,:) = exp(-(k/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*sum((cholPrec{c}*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
end
end

% total density summed over all components + Rao-Blackwellized component indicator
function [RBindicator,totalSampDens] = CombineMixtureComponents(mixWeights,sampDensPerComp)
nrSamples=size(sampDensPerComp,2);
nrComponents=size(sampDensPerComp,1);
RBindicator = sampDensPerComp.*repmat(mixWeights',1,nrSamples);
totalSampDens = sum(RBindicator);
RBindicator = RBindicator./repmat(totalSampDens,nrComponents,1);
end

% derivatives of Rao-Blackwellized component indicator
function [grb,hrb] = DerivativesLogRBind(x,RBindicator,mixMeans,mixPrecs,c)

% gradient
gn = zeros(size(x));
for j=1:length(mixMeans)
    gn = gn + RBindicator(j)*(mixPrecs{j}*(mixMeans{j}-x));
end
grb = mixPrecs{c}*(mixMeans{c}-x) - gn;

% hessian
hrb = -mixPrecs{c};
for j=1:length(mixMeans)
    hrb = hrb + RBindicator(j)*mixPrecs{j} - RBindicator(j)*(mixPrecs{j}*(mixMeans{j}-x)-gn)*(mixPrecs{j}*(mixMeans{j}-x))';
end

end