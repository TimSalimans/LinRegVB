% Test of inference methods for the probit model

% initial random seed, use to make sure the different methods are tested on
% the same simulated data
seed = 12345;

% select method
method = 'Hessian'; % 'VBEM', 'Basic', 'Factors', 'FactorGradient', 'Hessian', 'HessianMinibatch'

% settings
nrpoints=100;
nrreg=5;
nrrepeats=500;
if strcmp(method,'Basic')
    iterations = round(exp(log(100):0.5:log(1000)));
elseif strcmp(method,'Factors')
    iterations = round(exp(log(50):0.5:log(1000)));
elseif strcmp(method,'FactorGradient')
    iterations = round(exp(log(2):0.5:log(500))); % this one uses antithetics
elseif strcmp(method,'HessianMinibatch')
    iterations = unique(round(exp(0:0.5:log(1000))));
else
    iterations = round(exp(log(2):0.5:log(1000)));
end

% result containers
nrIncrement=length(iterations);
mse=zeros(nrIncrement,1);
logscore=zeros(nrIncrement,1);
time=zeros(nrIncrement,1);

% prior and initialization
priorMeanTimesPrec = zeros(nrreg,1);
priorPrec = eye(nrreg);
guessMean = zeros(nrreg,1);
guessPrec = eye(nrreg);

% do tests
for j=1:nrrepeats
    disp(['Running repeat #' int2str(j)])
    
    % set seed
    rng(seed);
    
    % simulate data
    beta = randn(nrreg,1);
    x = randn(nrpoints,nrreg)/sqrt(nrreg/3);
    y = x*beta+randn(nrpoints,1)>0;
    signedx = x.*repmat(2*y-1,1,nrreg);
    
    % save seed
    seed = rng;
    
    % try different numbers of iterations
    for i=1:nrIncrement
        
        % set seed
        rng(seed);
        
        % do inference using selected method, keep time
        switch method
            
            case 'VBEM'
                
                % do inference
                tic
                [meanv,prec] = ProbitVBEM(x,y,priorMeanTimesPrec,priorPrec,iterations(i));
                time(i)=time(i)+(toc/nrrepeats);
                
            case 'Basic'
                
                % define model
                logpostfun=@(b)(priorMeanTimesPrec'*b - 0.5*sum((priorPrec*b).*b) + sum(log(erfc((signedx*b)./-sqrt(2)))));
                
                % do inference
                tic
                [meanv,prec] = GaussVarApproxBasic(guessMean,guessPrec,logpostfun,iterations(i));
                time(i)=time(i)+(toc/nrrepeats);
                
            case 'Factors'
                
                % define likelihood function
                llhfun=@(z)log(erfc(z.*(2*y-1)./-sqrt(2)));
                
                % do inference
                tic
                [meanv,prec] = GaussVarApproxFactorGLM(priorMeanTimesPrec,priorPrec,x,llhfun,iterations(i));
                time(i)=time(i)+(toc/nrrepeats);
                
            case 'FactorGradient'
                
                % gradient of log-likelihood
                llhGrad = @(f)(repmat(2*y-1,1,size(f,2)).*(sqrt(2/pi)*exp(-0.5*(f.*repmat(2*y-1,1,size(f,2))).^2)./...
                    erfc(f.*repmat(2*y-1,1,size(f,2))./-sqrt(2))));
                
                % do inference
                tic
                [meanv,prec] = GaussVarApproxFactorGradientGLM(priorMeanTimesPrec,priorPrec,x,llhGrad,iterations(i));
                time(i)=time(i)+(toc/nrrepeats);
                
            case 'Hessian'
                
                % define model
                gradfun=@(b)ProbitGradHess(signedx,priorMeanTimesPrec,priorPrec,b);
                
                % do inference
                tic
                [meanv,prec] = GaussVarApproxHessian(guessMean,guessPrec,gradfun,iterations(i));
                time(i)=time(i)+(toc/nrrepeats);
                
            case 'HessianMinibatch'
                
                % define model
                gradfun=@(b,batchInd)ProbitGradHess(signedx,priorMeanTimesPrec,priorPrec,b,batchInd);
                
                % do inference
                tic
                [meanv,prec] = GaussVarApproxHessianMinibatch(guessMean,guessPrec,gradfun,iterations(i));
                time(i)=time(i)+(toc/nrrepeats);
                
            otherwise
                error('invalid inference method selected')
        end
        
        % calculate different measures of accuracy
        mse(i) = mse(i) + mean((beta - meanv).^2)/nrrepeats;
        halflogdet = sum(log(diag(chol(prec))));
        logprob = halflogdet - 0.5*nrreg*log(2*pi) - 0.5*(beta-meanv)'*prec*(beta-meanv);
        logscore(i) = logscore(i) + logprob/nrrepeats;
    end
end

% make plots

% specify line color
if strcmp(method,'Hessian')
    ls='-r';
elseif strcmp(method,'HessianMinibatch')
    ls='-m';
elseif strcmp(method,'VBEM')
    ls='-k';
elseif strcmp(method,'Basic')
    ls='-g';
elseif strcmp(method,'Factors')
    ls='-c';
elseif strcmp(method,'FactorGradient')
    ls='-b';
    iterations = iterations*2; % this one uses antithetics
end

% rmse
figure(1);
semilogx(iterations,sqrt(mse),ls);
xlabel('number of likelihood evaluations')
ylabel('RMSE approximate posterior mean')
hold('on');
%
% % log score
% figure(2);
% semilogx(iterations,logscore,ls);
% xlabel('number of likelihood evaluations')
% ylabel('number of likelihood evaluations')
% hold('on');