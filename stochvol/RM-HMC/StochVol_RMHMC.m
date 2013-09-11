function [] = StochVol_RMHMC()

% Random Numbers...
randn('state', sum(100*clock));
rand('twister', sum(100*clock));

% % True Parameters
% beta  = 0.65;
% sigma = 0.15;
% phi   = 0.98;
% 
% 
% %%%%% Generate Data %%%%%
% %{
% NumOfObs        = 2000;
% 
% x = zeros(NumOfObs,1); % Latent Volatilities
% y = zeros(NumOfObs,1); % Observed Data
% I = sparse(eye(NumOfObs));
% 
% % Generate latent volatility at t = 1
% x(1) = normrnd(0, sigma/sqrt(1 - phi^2));
% 
% % Generate latent volatilities
% for n = 1:(NumOfObs-1)
%     x(n + 1) = phi*x(n) + normrnd(0, sigma);
% end
% 
% % Generate observed data
% y = beta*randn(NumOfObs,1).*exp(x/2);
% 
% Truex = x; % Save true x values
% %}
% 
% 
% %%%%% Load Saved Data %%%%%
% %
% Data  = load('StochVolData1.mat');
% x     = Data.Truex;
% Truex = Data.Truex;
% y     = Data.y;
% NumOfObs = length(y);
% %}

Data  = load('../PoundDollar.mat');
y     = Data.y;
NumOfObs = length(y);

% Sparse diagonal matrix
I = sparse(eye(NumOfObs));

% Preallocate off diagonal elements in sparse matrix for speed
ITri = I;
ITri(2:NumOfObs+1:end)          = 0.1; % Subdiagonal
ITri(NumOfObs+1:NumOfObs+1:end) = 0.1; % Superdiagonal

% Precompute for speed
ySquared    = y.^2;


% RM-HMC Setup
NumOfIterations    = 110000;
BurnIn             = 10000;


NumOfLeapFrogSteps = 100;
Dist               = 8;
StepSize           = Dist/NumOfLeapFrogSteps;


NumOfHPLeapFrogSteps = 10;
HPDist               = 5;
HPStepSize           = HPDist/NumOfHPLeapFrogSteps;
HPNumOfFixPointSteps = 10;


Proposed = 0;
Accepted = 0;

HPProposed = 0;
HPAccepted = 0;


% Set initial values of x, beta, sigma, phi
x     = y;
beta  = 0.5;
sigma = 0.5;
phi   = 0.5;

% Preallocate space for saving results
betaSaved  = zeros(NumOfIterations-BurnIn,1);
phiSaved   = zeros(NumOfIterations-BurnIn,1);
sigmaSaved = zeros(NumOfIterations-BurnIn,1);
LJLSaved   = zeros(NumOfIterations-BurnIn,1);
HSaved     = zeros(NumOfIterations-BurnIn,1);



for IterationNum = 1:NumOfIterations

    
    if mod(IterationNum,100) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        drawnow
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample the latent variables %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Update the log-jointlikelihood
    CurrentLJL =  -(1/(2*sigma^2))*x(1)^2*(1-phi^2) - sum( x./2 + (ySquared)./(2*beta^2*exp(x)) ) - (1/(2*sigma^2))*sum( (x(2:end)-phi*x(1:end-1)).^2 );
    
    xNew = x;
    Proposed = Proposed + 1;
    
    % Create Gradient & Metric Tensor
    s   = -0.5 + (ySquared)./(2*beta^2*exp(xNew));
    d_1 = (1/sigma^2)*(xNew(1) - phi*xNew(2));
    d_2 = (1/sigma^2)*(xNew(end) - phi*xNew(end-1));
    r   = [d_1; -(phi/sigma^2)*(xNew(3:end) - phi*xNew(2:end-1)) + (1/sigma^2)*(xNew(2:end-1) - phi*xNew(1:end-2)); d_2];

    % Gradient
    GradL = s - r;
    
    iC                            = ITri;
    iC(1:NumOfObs+1:end)          = (1+phi^2)/sigma^2;
    iC(1,1)                       = 1/sigma^2;
    iC(end,end)                   = 1/sigma^2;
    iC(2:NumOfObs+1:end)          = -phi/sigma^2; % Lower off diagonal
    iC(NumOfObs+1:NumOfObs+1:end) = -phi/sigma^2; % Upper off diagonal
    
    G                   = iC;
    G(1:NumOfObs+1:end) = G(1:NumOfObs+1:end) + 0.5; % Add diag of 0.5 i.e. Fisher information
    
    CholG = chol(G);
    
    ProposedMomentum = ( randn(1,NumOfObs)*CholG )';
    OriginalMomentum = ProposedMomentum;
        
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
        
    RandomSteps = ceil(rand*NumOfLeapFrogSteps);
        
    % Perform leapfrog steps
    for StepNum = 1:RandomSteps
            
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        % Log det term is zero!
        % Exact since constant metric tensor
        ProposedMomentum = ProposedMomentum + TimeStep*(StepSize/2)*( GradL );
        

        %%%%%%%%%%%%%%%%%%%%%%%
        % Update w parameters %
        %%%%%%%%%%%%%%%%%%%%%%%
        % Exact since constant metric tensor
        xNew = xNew + (TimeStep*StepSize)*(G\ProposedMomentum);            
        
        % Create Gradient
        s   = (ySquared)./(2*beta^2*exp(xNew)) - 0.5;
        d_1 = (1/sigma^2)*(xNew(1) - phi*xNew(2));
        d_2 = (1/sigma^2)*(xNew(end) - phi*xNew(end-1));
        r   = [d_1; -(phi/sigma^2)*(xNew(3:end) - phi*xNew(2:end-1)) + (1/sigma^2)*(xNew(2:end-1) - phi*xNew(1:end-2)); d_2];
        
        % Gradient
        GradL = s - r;
        

        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        % Log det term is zero!
        % Exact since constant metric tensor
        ProposedMomentum = ProposedMomentum + TimeStep*(StepSize/2)*( GradL );
            
    end

    
    % Calculate proposed H value
    ProposedLJL =  -(1/(2*sigma^2))*xNew(1)^2*(1-phi^2) - sum( xNew./2 + (ySquared)./(2*beta^2*exp(xNew)) ) - (1/(2*sigma^2))*sum( (xNew(2:end)-phi*xNew(1:end-1)).^2 );
    ProposedLogDet = 0.5*( 2*sum(log(diag(CholG))) );
    ProposedH      = -ProposedLJL + ProposedLogDet + ((ProposedMomentum'/G)*ProposedMomentum)/2;
    
        
    % Calculate current H value
    CurrentLogDet = 0.5*( 2*sum(log(diag(CholG))) ); % G is fixed...
    CurrentH      = -CurrentLJL + CurrentLogDet + ((OriginalMomentum'/G)*OriginalMomentum)/2; % G is fixed...

    
    % Accept according to ratio
    Ratio = -ProposedH + CurrentH;

    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        x = xNew;
        Accepted = Accepted + 1;
        %disp('Accepted')
        %drawnow
        CurrentH = ProposedH;
    else
        %disp('Rejected')
        %drawnow
    end
    
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample the parameters %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Precalculate exp(x) for speed
    Expx = exp(x);
    
    % Update the log-jointlikelihood
    CurrentLJL = -sum(x/2) - NumOfObs*log(beta) - sum( (ySquared)./(2*beta^2*Expx) ) + 0.5*log(1-phi^2) - log(sigma) - x(1)^2*(1-phi^2)/(2*sigma^2) - (NumOfObs-1)*log(sigma) - sum( (x(2:end)-phi*x(1:end-1)).^2./(2*sigma^2) );
    CurrentJ   = log(sigma*(1-phi^2)); % Jacobian
    % Add Prior
    Prior      = -beta - 0.5/(2*sigma^2) - 6*log(sigma^2) + log(sigma) + 19*log((phi+1)/2) + 0.5*log((1-phi)/2);
    CurrentLJL = CurrentLJL + Prior;
    
    HPGradL(1) = -NumOfObs/beta + sum( (ySquared)./(beta^3*Expx) );
    HPGradL(2) = -NumOfObs + x(1)^2*(1-phi^2)/(sigma^2) + sum( ((x(2:end)-phi*x(1:end-1)).^2)/(sigma^2) );
    HPGradL(3) = -phi + (phi*x(1)^2)*(1-phi^2)/sigma^2 + sum( x(1:end-1).*(x(2:end)-phi*x(1:end-1)).*(1-phi^2)./sigma^2 );
    % Add Prior
    HPGradL(1) = HPGradL(1) - 1;
    HPGradL(2) = HPGradL(2) + 0.5/sigma^2 - 11;
    HPGradL(3) = HPGradL(3) + 38*(1-phi) - (1+phi);
    
    HPNew      = [beta sigma phi];
    HPProposed = HPProposed + 1;
    
    
    % Fisher Information & Metric Tensor
    G = zeros(3);
    G(1,1) = (2*NumOfObs)/beta^2;
    G(2,2) = (2*NumOfObs);
    G(2,3) = 2*phi;
    G(3,2) = G(2,3);
    G(3,3) = 2*phi^2 - (NumOfObs-1)*(phi^2 - 1);
    % Add Prior
    PriorG = zeros(3);
    PriorG(2,2) = -1/sigma^2;
    PriorG(3,3) = (-38-1)*(1-phi^2);
    
    G = G - PriorG;
    
    OriginalCholG = chol(G + eye(3)*1e-6);
    OriginalG     = G;
    
    HPProposedMomentum = ( randn(1,3)*OriginalCholG )';
    HPOriginalMomentum = HPProposedMomentum;
    
    % w.r.t. beta
    dGdParas{1} = zeros(3);
    dGdParas{1}(1,1) = (-4*NumOfObs)/beta^3;
    % w.r.t. sigma
    dGdParas{2} = zeros(3);
    % Add Prior w.r.t. sigma
    dGdParas{2}(2,2) = dGdParas{2}(2,2) - (2/sigma^2);
    % w.r.t. phi
    dGdParas{3} = zeros(3);
    dGdParas{3}(2,3) = 2*(1-phi^2);
    dGdParas{3}(3,2) = dGdParas{3}(2,3);
    dGdParas{3}(3,3) = ((4*phi) - (NumOfObs-1)*(2*phi))*(1-phi^2);
    % Add Prior w.r.t. phi
    dGdParas{3}(3,3) = dGdParas{3}(3,3) - (4*phi)*(20+1.5-2)*(1-phi^2);
    
    InvGdG{1} = G\dGdParas{1};
    InvGdG{2} = G\dGdParas{2};
    InvGdG{3} = G\dGdParas{3};
    
    TraceInvGdG(1) = trace(InvGdG{1});
    TraceInvGdG(2) = trace(InvGdG{2});
    TraceInvGdG(3) = trace(InvGdG{3});
    
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
        
    RandomSteps = ceil(rand*NumOfHPLeapFrogSteps);
        
    % Perform leapfrog steps
    for StepNum = 1:RandomSteps
            
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        
        TraceTerm = 0.5*TraceInvGdG';
            
        % Multiple fixed point iteration
        PM = HPProposedMomentum;
        for FixedIter = 1:HPNumOfFixPointSteps

            InvGMomentum = G\PM;
            
            LastTerm(1)  = 0.5*(PM'*InvGdG{1}*InvGMomentum);
            LastTerm(2)  = 0.5*(PM'*InvGdG{2}*InvGMomentum);
            LastTerm(3)  = 0.5*(PM'*InvGdG{3}*InvGMomentum);

            PM = HPProposedMomentum + TimeStep*(HPStepSize/2)*(HPGradL' - TraceTerm + LastTerm');
        end
        HPProposedMomentum = PM;
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%
        % Update w parameters %
        %%%%%%%%%%%%%%%%%%%%%%%
        
        PHP = HPNew;
        
        OriginalPHP    = HPNew;
        OriginalPHP(2) = log(OriginalPHP(2));
        OriginalPHP(3) = atanh(OriginalPHP(3));
        for FixedIter = 1:HPNumOfFixPointSteps

            InvGMomentum = G\HPProposedMomentum;
            
            if FixedIter == 1
                OriginalInvGMomentum = InvGMomentum;
            end
            
            PHP(2) = log(PHP(2));
            PHP(3) = atanh(PHP(3));
            
            PHP = OriginalPHP + (TimeStep*(HPStepSize/2))*OriginalInvGMomentum' + (TimeStep*(HPStepSize/2))*InvGMomentum';
            
            PHP(2) = exp(PHP(2));
            PHP(3) = tanh(PHP(3));
            
            % Calculate G
            G = zeros(3);
            G(1,1) = (2*NumOfObs)/PHP(1)^2;
            G(2,2) = (2*NumOfObs);
            G(2,3) = 2*PHP(3);
            G(3,2) = G(2,3);
            G(3,3) = 2*PHP(3)^2 - (NumOfObs-1)*(PHP(3)^2 - 1);
            % Add Prior
            PriorG = zeros(3);
            PriorG(2,2) = -1/PHP(2)^2;
            PriorG(3,3) = (-38-1)*(1-PHP(3)^2);
            G = G - PriorG;        
        end
        HPNew = PHP;
        
            
        HPGradL(1) = -NumOfObs/HPNew(1) + sum( (ySquared)./(HPNew(1)^3*Expx) );
        HPGradL(2) = -NumOfObs + x(1)^2*(1-HPNew(3)^2)/(HPNew(2)^2) + sum( ((x(2:end)-HPNew(3)*x(1:end-1)).^2)/(HPNew(2)^2) );
        HPGradL(3) = -HPNew(3) + (HPNew(3)*x(1)^2)*(1-HPNew(3)^2)/HPNew(2)^2 + sum( x(1:end-1).*(x(2:end)-HPNew(3)*x(1:end-1)).*(1-HPNew(3)^2)./HPNew(2)^2 );
        % Add Prior
        HPGradL(1) = HPGradL(1) - 1;
        HPGradL(2) = HPGradL(2) + 0.5/HPNew(2)^2 - 11;
        HPGradL(3) = HPGradL(3) + 38*(1-HPNew(3)) - (1+HPNew(3));
        

        % w.r.t. beta
        dGdParas{1} = zeros(3);
        dGdParas{1}(1,1) = (-4*NumOfObs)/HPNew(1)^3;
        % w.r.t. sigma
        dGdParas{2} = zeros(3);
        % Add Prior w.r.t. sigma
        dGdParas{2}(2,2) = dGdParas{2}(2,2) - (2/HPNew(2)^2);
        % w.r.t. phi
        dGdParas{3} = zeros(3);
        dGdParas{3}(2,3) = 2*(1-HPNew(3)^2);
        dGdParas{3}(3,2) = dGdParas{3}(2,3);
        dGdParas{3}(3,3) = ((4*HPNew(3)) - (NumOfObs-1)*(2*HPNew(3)))*(1-HPNew(3)^2);
        % Add Prior w.r.t. phi
        dGdParas{3}(3,3) = dGdParas{3}(3,3) - (4*HPNew(3))*(20+1.5-2)*(1-HPNew(3)^2);
        
        InvGdG{1} = G\dGdParas{1};
        InvGdG{2} = G\dGdParas{2};
        InvGdG{3} = G\dGdParas{3};
        
        TraceInvGdG(1) = trace(InvGdG{1});
        TraceInvGdG(2) = trace(InvGdG{2});
        TraceInvGdG(3) = trace(InvGdG{3});
        
                
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        TraceTerm = 0.5*TraceInvGdG';
        
        InvGMomentum = G\HPProposedMomentum;
        
        LastTerm(1)  = 0.5*(HPProposedMomentum'*InvGdG{1}*InvGMomentum);
        LastTerm(2)  = 0.5*(HPProposedMomentum'*InvGdG{2}*InvGMomentum);
        LastTerm(3)  = 0.5*(HPProposedMomentum'*InvGdG{3}*InvGMomentum);
        
        HPProposedMomentum = HPProposedMomentum + TimeStep*(HPStepSize/2)*(HPGradL' - TraceTerm + LastTerm');
        
    end

    
        
    % Calculate proposed H value
    ProposedLJL = -sum(x/2) - NumOfObs*log(HPNew(1)) - sum( (ySquared)./(2*HPNew(1)^2*exp(x)) ) + 0.5*log(1-HPNew(3)^2) - log(HPNew(2)) - x(1)^2*(1-HPNew(3)^2)/(2*HPNew(2)^2) - (NumOfObs-1)*log(HPNew(2)) - sum( (x(2:end)-HPNew(3)*x(1:end-1)).^2./(2*HPNew(2)^2) );
    ProposedJ   = log(HPNew(2)*(1-HPNew(3)^2)); % Jacobian
    % Add Prior
    Prior       = -HPNew(1) - 0.5/(2*HPNew(2)^2) - 6*log(HPNew(2)^2) + log(HPNew(2)) + 19*log((HPNew(3)+1)/2) + 0.5*log((1-HPNew(3))/2);
    ProposedLJL = ProposedLJL + Prior;
    
    try
        ProposedLogDet = 0.5*( 2*sum(log(diag( chol(G+eye(3)*1e-6) ))) );
    catch
        disp('error')
    end
    
    ProposedH     = -ProposedLJL - ProposedJ + ProposedLogDet + ((HPProposedMomentum'/G)*HPProposedMomentum)/2;
    
    % Calculate current H value
    CurrentLogDet = 0.5*( 2*sum(log(diag(OriginalCholG))) );
    CurrentH      = -CurrentLJL - CurrentJ + CurrentLogDet + ((HPOriginalMomentum'/OriginalG)*HPOriginalMomentum)/2;
    
    % Accept according to ratio
    Ratio = -ProposedH + CurrentH;

    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        beta  = HPNew(1);
        sigma = HPNew(2);
        phi   = HPNew(3);
        HPAccepted = HPAccepted + 1;
        %disp('Accepted')
        %drawnow
        CurrentH = ProposedH;
    else
        %disp('Rejected')
        %drawnow
    end
    
    
    
    
    
    if mod(IterationNum, 100) == 0
        Acceptance   = Accepted/Proposed;
        HPAcceptance = HPAccepted/HPProposed;
        
        if mod(IterationNum, 100) == 0
            disp(['HP:' num2str(HPAcceptance)])
            disp(['Latent volatilities:' num2str(Acceptance)])
        end
        
        Proposed = 0;
        Accepted = 0;
        
        HPProposed = 0;
        HPAccepted = 0;
    end
        

    % Save samples if required
    if IterationNum > BurnIn
        betaSaved(IterationNum-BurnIn,1)  = beta;
        phiSaved(IterationNum-BurnIn,1)   = phi;
        sigmaSaved(IterationNum-BurnIn,1) = sigma;
        LJLSaved(IterationNum-BurnIn)     = CurrentLJL;
        HSaved(IterationNum-BurnIn)       = CurrentH;
    end
    
    % Start timer after burn-in
    if IterationNum == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
        tic;
    end
    
end

% Stop timer
TimeTaken = toc;


CurTime = fix(clock);
%save(['Results/RMHMC_Trans_' num2str(HPNumOfFixPointSteps) '_FixPointIts_' num2str(Dist) '_' num2str(NumOfLeapFrogSteps) '_' num2str(HPDist) '_' num2str(NumOfHPLeapFrogSteps) '_StochVol_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'HSaved', 'y', 'betaSaved', 'phiSaved', 'sigmaSaved', 'LJLSaved', 'TimeTaken', 'NumOfLeapFrogSteps', 'StepSize', 'NumOfHPLeapFrogSteps', 'HPStepSize');
save Results/RMHMC_sample.mat betaSaved phiSaved sigmaSaved

figure(100)
subplot(231)
plot(betaSaved)
subplot(232)
plot(sigmaSaved)
subplot(233)
plot(phiSaved)
% Plot histograms
%
subplot(234)
hist(betaSaved,1000)
subplot(235)
hist(sigmaSaved,1000)
subplot(236)
hist(phiSaved,1000)
%}

end


