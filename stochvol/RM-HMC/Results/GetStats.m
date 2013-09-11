Files = dir('RMHMC_Trans*.*');

Data = [];

for i = 1:length(Files)
    disp(i)
    
    clear Data
    
    % Open file
    Data = open(Files(i).name);
    
    TimeTaken(i) = Data.TimeTaken;
    
    betaESS(i)  = CalculateESS(Data.betaSaved, 19999);
    sigmaESS(i) = CalculateESS(Data.sigmaSaved, 19999);
    phiESS(i)   = CalculateESS(Data.phiSaved, 19999);
    
    %
    for j = 1:length(Data.xSaved(1,:))
        disp([num2str(i) ' ' num2str(j)])
        xESS(i,j) = CalculateESS(Data.xSaved(:,j), 19999);
    end
    %}
end

save('ESS_Results.mat', 'betaESS', 'sigmaESS', 'phiESS', 'xESS', 'TimeTaken')

TimeTakenMean = mean(TimeTaken); 
betaMean  = mean(betaESS);
sigmaMean = mean(sigmaESS);
phiMean   = mean(phiESS);
xMean     = mean(xESS,1);

disp(['Mean time taken: ' num2str(TimeTakenMean)]);
disp(['Mean beta ESS: ' num2str(betaMean)]);
disp(['Mean sigma ESS: ' num2str(sigmaMean)]);
disp(['Mean phi ESS: ' num2str(phiMean)]);
disp(['(Min,Median,Max) x ESS: (' num2str(min(xMean)) ',' num2str(median(xMean)) ',' num2str(max(xMean)) ')'])