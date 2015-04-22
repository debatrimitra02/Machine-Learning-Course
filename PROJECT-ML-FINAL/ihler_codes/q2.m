clear all;
X=load('data/kaggle.X1.train.txt');
Y=load('data/kaggle.Y.train.txt');

X_TEST=load('data/kaggle.X1.test.txt');
%[Xtr,Xte,Ytr,Yte] = crossValidate(X,Y,5,3);


%dt = treeRegress(Xtr,Ytr, 'maxDepth',20);

%Yte = predict(dt,Xte);
%e=mse(dt,Xte,Yte)

error=zeros(1,20);
for j=1:5
    [Xtr,Xte,Ytr,Yte] = splitData(X, Y, 0.6);
for i=1:20
  dt = treeRegress(Xtr,Ytr, 'maxDepth',i);
  %Yte = predict(dt,Xte);
  error(i)=error(i)+mse(dt,Xte,Yte);
end
end
error=error./5;
plot(error)

Error=zeros(1,10);
for j=1:5
    [X,Y]=shuffleData(X,Y);
    [Xtr,Xte,Ytr,Yte] = splitData(X, Y, 0.6);
for i=3:12
  dt = treeRegress(Xtr,Ytr, 'maxDepth',20,'minParent',2^i);
  %Yte = predict(dt,Xte);
  Error(i-2)=mse(dt,Xte,Yte);
end
end
Error=Error./5;
plot(Error)

dt = treeRegress(X,Y, 'maxDepth',10,'minParent',1000);
Y_TEST=predict(dt,X_TEST);