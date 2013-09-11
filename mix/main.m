% reproduce the plots in the paper

clc

% settings
nrSteps=1e3;

% load data
load cancermortality
y=double(cancermortality.y);
n=double(cancermortality.n);

% define log likelihood
logPostDens = @(x)BetaBinomialLogPosterior(x,y,n);
logPostDensQuad = @(x1,x2)BetaBinomialLogPosteriorQuad(x1,x2,y,n);
gradfun = @(x)BetaBinomialGradFun(x,y,n);

% contour grid
[theta1,theta2] = meshgrid(-7.7:.02:-5.6,4.5:0.1:14);
cvals = [-0.15 -0.5 -1 -2 -3 -5];
Z=zeros(size(theta1));

% contour plot true density
disp('Plotting true density')
figure()
subplot(3,3,9)
for i=1:size(theta1,1)
    for j=1:size(theta1,2)
        Z(i,j)=logPostDens([theta1(i,j);theta2(i,j)]);
    end
end
mzr = max(max(Z));
normconst = dblquad(@(x1,x2)exp(logPostDensQuad(x1,x2)-mzr),-10,-3.5,2,20);
Z=Z-mzr-log(normconst);
[C,h] = contour(theta1,theta2,Z,cvals);
colormap jet
xlabel('logit m');
ylabel('log K');
title('exact')

% set up KL divergence stuff
normPostDens = @(x1,x2)exp(logPostDensQuad(x1,x2)-mzr-log(normconst));
KLdiv=zeros(1,8);
KLdiv2=zeros(1,8);
R2=zeros(1,8);

% try multiple mixture components
for i=1:8
    
    disp(['Estimating approximation with ' int2str(i) ' components.'])
    nrComponents=i;
    
    if i>1
        
        % fit approximation
        [mixWeights,mixMeans,mixPrecs] = MixVarApprox(logPostDens,gradfun,nrComponents,nrSteps,[-7; 8]);
        
        % approximation density
        myDens=@(x1,x2)DensApproximation(x1,x2,mixWeights,mixMeans,mixPrecs);
        myLogDens=@(x1,x2)log(myDens(x1,x2));
        
    else
        % fitted using approximation with a single Gaussian, see the probit code
        approxMean = [-6.8240; 7.8287];
        approxPrec = [18.1893,    1.7892;  1.7892,    1.0084];
                
        % contour plot approximation
        myLogDens=@(x1,x2)LogDensApproximation2(x1,x2,approxMean,approxPrec);
        myDens=@(x1,x2)exp(myLogDens(x1,x2));
    end
    
    
    % calculate divergence
    KLdiv(i) = CalcKLdiv(myDens,normPostDens,-10,-3.5,2,20);
    
    % calculate divergence approximation
    KLdiv2(i) = CalcKLdiv2(myDens,normPostDens,-10,-3.5,2,20);
    
    % calculate R-squared
    R2(i)=CalcRsquared(myDens,normPostDens,-10,-3.5,2,20);
    
    % contour plot approximation
    subplot(3,3,i)
    for l=1:size(theta1,1)
        for j=1:size(theta1,2)
            Z(l,j)=myLogDens(theta1(l,j),theta2(l,j));
        end
    end
    [C,h] = contour(theta1,theta2,Z,cvals);
    colormap jet
    xlabel('logit m');
    ylabel('log K');
    title(int2str(i))
end

% plot KL-divergence + approximation + R-squared
figure()
plot(1:8,KLdiv,'-')
hold on
plot(1:8,KLdiv2,'--')
xlabel('number of mixture components');
ylabel('KL divergence D(q|p)');
figure()
plot(1:8,R2)
xlabel('number of mixture components');
ylabel('R-squared');
disp(['final R-squared: ' num2str(R2(end))]);
