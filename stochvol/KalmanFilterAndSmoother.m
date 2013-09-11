% approximate posterior moments for the volatitilies via Kalman filter and smoother
function [meanM,meanX,varM,varX,covMX] = KalmanFilterAndSmoother(xtau,xprec,varDraws,phiDraws) %#codegen

% dimension
nrsamples=length(varDraws);
nrvols=length(xtau);

% pre-allocate storage for Kalman filter + smoother
meanM=zeros(nrvols,nrsamples);
meanX=zeros(nrvols,nrsamples);
varM=zeros(nrvols,nrsamples);
varX=zeros(nrvols,nrsamples);
covMX=zeros(nrvols,nrsamples);
v=zeros(nrvols,1);
F=zeros(nrvols,1);
K1=zeros(nrvols,1);
K2=zeros(nrvols,1);
L11=zeros(nrvols,1);
L12=zeros(nrvols,1);
L21=zeros(nrvols,1);
L22=zeros(nrvols,1);
N11=zeros(nrvols+1,1);
N12=zeros(nrvols+1,1);
N21=zeros(nrvols+1,1);
N22=zeros(nrvols+1,1);
r1=zeros(nrvols+1,1);
r2=zeros(nrvols+1,1);

% use Kalman filter to get marginal likelihoods (vols integrated out)
% + Kalman smoother to calculate natural gradient for likelihood approximation
for i=1:nrsamples
    
    % initialize Kalman filter
    varM(1,i)=1e10; % should really be infinite, but something large will do
    varX(1,i)=varDraws(i)/(1-phiDraws(i)^2);
    
    % filter
    for t=1:nrvols
        
        % prediction error & predictive variance
        v(t)=xtau(t)/xprec(t) - meanM(t,i) - meanX(t,i); 
        F(t)=varM(t,i)+varX(t,i)+2*covMX(t,i)+1/xprec(t);
        
        % Kalman gain
        K1(t)=(varM(t,i)+covMX(t,i))/F(t);
        K2(t)=phiDraws(i)*(varX(t,i)+covMX(t,i))/F(t);
        
        % some intermediate quantities
        L11(t)=1-K1(t);
        L12(t)=-K1(t);
        L21(t)=-K2(t);
        L22(t)=phiDraws(i)-K2(t);
        
        % update & predict if there are more periods
        if t<nrvols
            meanM(t+1,i)=meanM(t,i)+K1(t)*v(t);
            meanX(t+1,i)=phiDraws(i)*meanX(t,i)+K2(t)*v(t);
            varM(t+1,i)=varM(t,i)*L11(t)+covMX(t,i)*L12(t);
            varX(t+1,i)=phiDraws(i)*(covMX(t,i)*L21(t)+varX(t,i)*L22(t))+varDraws(i);
            covMX(t+1,i)=varM(t,i)*L21(t)+covMX(t,i)*L22(t);
        end
    end
    
    % smooth
    for t=nrvols:-1:1
        
        % calculate quantities for backward recursion
        nt=t+1;
        NL11=N11(nt)*L11(t)+N12(nt)*L21(t);
        NL12=N11(nt)*L12(t)+N12(nt)*L22(t);
        NL21=N21(nt)*L11(t)+N22(nt)*L21(t);
        NL22=N21(nt)*L12(t)+N22(nt)*L22(t);
        
        N11(t)=L11(t)*NL11+L21(t)*NL21 + 1/F(t);
        N12(t)=L11(t)*NL12+L21(t)*NL22 + 1/F(t);
        N21(t)=L12(t)*NL11+L22(t)*NL21 + 1/F(t);
        N22(t)=L12(t)*NL12+L22(t)*NL22 + 1/F(t);
        
        r1(t)=v(t)/F(t) + L11(t)*r1(nt) + L21(t)*r2(nt);
        r2(t)=v(t)/F(t) + L12(t)*r1(nt) + L22(t)*r2(nt);
        
        NP11=N11(t)*varM(t,i)+N12(t)*covMX(t,i);
        NP12=N11(t)*covMX(t,i)+N12(t)*varX(t,i);
        NP21=N21(t)*varM(t,i)+N22(t)*covMX(t,i);
        NP22=N21(t)*covMX(t,i)+N22(t)*varX(t,i);
        
        % smooth posterior moments M and X
        meanM(t,i)=meanM(t,i) + varM(t,i)*r1(t) + covMX(t,i)*r2(t);
        meanX(t,i)=meanX(t,i) + covMX(t,i)*r1(t) + varX(t,i)*r2(t);
        
        postVarM=varM(t,i)-varM(t,i)*NP11-covMX(t,i)*NP21;
        postVarX=varX(t,i)-covMX(t,i)*NP12-varX(t,i)*NP22;
        postCovMX=covMX(t,i)-covMX(t,i)*NP11-varX(t,i)*NP21;
        
        varM(t,i)=postVarM;
        varX(t,i)=postVarX;
        covMX(t,i)=postCovMX;
    end
end
