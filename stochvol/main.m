% do everything

disp('Compiling Kalman filter + smoother')
DoMex
disp('Approximating the posterior of the stochastic volatility model')
InferStochVol
disp('Running Girolami et al.''s MCMC sampler')
cd('RM-HMC')
StochVol_RMHMC
cd('..')
disp('Making the plots')
MakePlots