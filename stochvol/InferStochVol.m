% run variational inference for the stochastic volatility model and compare
% against Monte Carlo

% clear screen
clc

% set seed
rng(12345,'twister')

% get data
load PoundDollar

% set the number of steps
nrSteps = 500;

% prior parameters in natural parameterization
priorParPhi=[20-1; 1.5-1];
priorParVar=[5-1; 0.25];

% do inference
tic
[postParPhi,postParVar,lhVolsMeanTimesPrec,lhVolsPrec] = ApproxStochVol(y.^2,priorParPhi,priorParVar,nrSteps);
toc

% save results
save LinRegRes.mat postParPhi postParVar lhVolsMeanTimesPrec lhVolsPrec