% make plots

% load results
load RM-HMC/Results/RMHMC_sample
load LinRegRes

% sample from approximation
nrOfSamples = 5000;
[phiDraws,varDraws] = SamplePhiAndVar(postParPhi,postParVar,nrOfSamples);

% get means and variances for Beta
[meanM,meanX,varM,varX,covMX] = KalmanFilterAndSmoother_mex(lhVolsMeanTimesPrec,lhVolsPrec,varDraws,phiDraws);

% adapt scale of M
meanM=meanM/2;
varM=varM/4;

% density function beta
pdfbeta = @(x)mean(exp(-0.5*(log(x)-meanM(1,:)).^2./varM(1,:))./(sqrt(2*pi*varM(1,:)).*x));

% plot beta
figure(1)
histnorm(betaSaved,1000)
hold on
minval=min(betaSaved);
maxval=max(betaSaved);
x=minval:((maxval-minval)/100):maxval;
y=zeros(size(x));
for i=1:length(x)
    y(i)=pdfbeta(x(i));
end
plot(x,y)
xlabel('beta');
ylabel('posterior density');

% plot phi
figure(2)
histnorm(phiSaved,1000)
hold on
minval=min(phiSaved);
maxval=max(phiSaved);
x=minval:((maxval-minval)/100):maxval;
y=betapdf((x+1)/2,postParPhi(1)+1,postParPhi(2)+1)/2; % make sure to account for the difference in parameterization
plot(x,y)
xlabel('phi');
ylabel('posterior density');

% plot var
varSaved = sigmaSaved.^2;
figure(3)
histnorm(varSaved,1000)
hold on
minval=min(varSaved);
maxval=max(varSaved);
x=minval:((maxval-minval)/100):maxval;
% make sure to account for the difference in parameterization
a = repmat(postParVar(1)+1,nrOfSamples,1);
b = postParVar(2)+postParVar(3)*phiDraws.^2;
logpdfs = -(a+1)*log(x)-b*(1./x) + repmat(a.*log(b) - gammaln(a),1,size(x,2));
mv = max(logpdfs);
logpdfs = logpdfs - repmat(mv,nrOfSamples,1);
y=mean(exp(logpdfs)).*exp(mv);
plot(x,y)
xlabel('variance');
ylabel('posterior density');
