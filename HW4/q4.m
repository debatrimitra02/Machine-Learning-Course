clear all;
X=load('data/kaggle.X1.train.txt');
Y=load('data/kaggle.Y.train.txt');
[Xtr,Xte,Ytr,Yte] = crossValidate(X,Y,5,3);
mu = mean(Ytr);

dY = Ytr - mu.*ones(size(Ytr,1),1);
Nboost=25;

for k=1:25,
    Learner{k} = treeRegress(Xtr,dY,'maxDepth',3,'minParent',32);
    %alpha(k) = 1; 
    dY = dY-  predict(Learner{k}, Xtr);
end

[Ntest,D0] = size(Xte);
[Ntrain,D]=size(Xtr);

prediction_tr = zeros(Ntrain,1);
prediction_te = zeros(Ntest,1);

%error=mse(Learner{5},Xte,Yte)
%j=[1,5,10,25];
error_train=zeros(4,1);
error_test=zeros(4,1);
i=1;
for j=[1,5,10,25]
    prediction_tr = zeros(Ntrain,1);
    prediction_te = zeros(Ntest,1);
    
    for k=1:j,
        prediction_tr = prediction_tr + predict(Learner{k}, Xtr);
    end;

    error_train(i) = mean( sum( (Ytr - prediction_tr).^2,2) );
    i=i+1;
end
i=1;
for  j=[1,5,10,25]
    prediction_tr = zeros(Ntrain,1);
    prediction_te = zeros(Ntest,1);
    for k=1:j,
        prediction_te = prediction_te + predict(Learner{k}, Xte);
    end;
    
    error_test(i) = mean( sum( (Yte - prediction_te).^2,2) );
    i=i+1;
end   
plot(error_test)
hold on
plot(error_train)
