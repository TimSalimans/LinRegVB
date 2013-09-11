% sample stoch vol model parameters
function [phiDraws,varDraws] = SamplePhiAndVar(postParPhi,postParVar,nrOfSamples)

% sample phi
betaDraws = betarnd(postParPhi(1)+1,postParPhi(2)+1,[nrOfSamples 1]);
phiDraws = 2*betaDraws-1;

% sample var | phi
gamDraws = gamrnd(postParVar(1)+1,1,[nrOfSamples 1]);
varDraws = (postParVar(2)+postParVar(3)*phiDraws.^2)./gamDraws;