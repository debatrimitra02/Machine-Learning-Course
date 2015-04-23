clear all;
X=load('data/kaggle.X1.train.txt');
Y=load('data/kaggle.Y.train.txt');

[X,Y] = shuffleData(X,Y); % permute out of class-based order
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,.6); % divide into training & test
[Xtr,S] = rescale(Xtr); 
Xte = rescale(Xte,S);

BaggedEnsemble = TreeBagger(50,Xtr,Ytr,'OOBPred','On','method','regression');
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
plot(oobErrorBaggedEnsemble)

error=mse(BaggedEnsemble,Xte,Yte);