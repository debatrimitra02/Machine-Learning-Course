clear all;
Xtrain=load('kaggle.X1.train.txt');
Ytrain =load('kaggle.Y.train.txt');
Xtest= load('kaggle.X1.test.txt');
[Xtr,Xte,Ytr,Yte] = splitData(Xtrain, Ytrain, 0.7);

%% ======== Bayesian MARS model for Gaussian response data ========= %%

% data=cat(2,Ytr,Xtr);
% test=cat(2,Yte,Xte);
% 
% [test_set_predictions, chain_stats] = bayes_mars_gauss(data,test);
% 
% [test_predictions] = make_mars_forecast(Xtest);

% Kaggle score = 0.62135 (rank 120/223)

%% ======== Support Vector Regression ========%%       
%svrobj = svr_trainer(Xtrain,Ytrain,400,0.000000025,'gaussian',0.5);        

%% while using PCA %% 

%x=dataset(:,1:end-1);
%y=dataset(:,end);
%B=zscore(Xtr);
%[coeff,score]=princomp(B);
%X=B*coeff;
%E=cumsum(var(SCORE))/sum(var(SCORE));
%X=X1(:,1:2);
C = corr(Xtrain,Xtrain);          %find the correlated features
Xtr_norm=mat2gray(Xtrain);

figure, imagesc(C),colorbar  %visualize correlated features

[wcoeff,score,latent,tsquared,explained] = pca(Xtr_norm,'VariableWeights','variance');
var_explained=cumsum(explained);
figure; plot(var_explained);