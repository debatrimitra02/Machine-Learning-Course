clear all;
X=load('data/kaggle.X1.train.txt');
Y=load('data/kaggle.Y.train.txt');

[X,Y] = shuffleData(X,Y); % permute out of class-based order
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,.6); % divide into training & test
[Xtr,S] = rescale(Xtr); 
Xte = rescale(Xte,S);


y0=zeros(size(Yte,1),20);
for b=1:20,
[Xb,Yb] = bootstrapData(Xtr,Ytr,size(Xtr,1));
boot{b} = treeClassify(Xb,Yb, 1,inf,0,100);
y0(:,b) = predict(boot{b},Xte);
er(b) = mean( (mean(y0(:,1:b),2)>.5)~= Yte);
end;
figure; plot(1:20,er,'g-','linewidth',3);




