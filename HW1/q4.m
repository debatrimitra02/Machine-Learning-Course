clear all;

iris=load('data/iris.txt'); 

y=iris(:,end); 
x=iris(:,1:2);
[x, y] = shuffleData(x,y);
[Xtr,Xte ,Ytr, Yte] = splitData(x,y, .75);

%% 4(a)
m=1;
n=1;
p=1;
for i=1:length(Xtr(:,1))
 if(Ytr(i)==0)
     Zero(m,:)=Xtr(i,:);
     m=m+1;
 elseif(Ytr(i)==1)
     One(n,:)= Xtr(i,:);   
     n=n+1;
     elseif(Ytr(i)==2)
     Two(p,:)= Xtr(i,:);
     p=p+1;
  end
end

MEAN(:,:,1)=mean(Zero);
MEAN(:,:,2)=mean(One);
MEAN(:,:,3)=mean(Two);
%COV=zeros(2,2,3);
COV(:,:,1)=cov(Zero);
COV(:,:,2)=cov(One);
COV(:,:,3)=cov(Two);

%% 4(b)
group=Ytr;
figure;
gscatter(Xtr(:,1),Xtr(:,2),group)
%% 4(c)
figure;
gscatter(Xtr(:,1),Xtr(:,2),group)
hold on;
plotGauss2D(MEAN(:,:,1),COV(:,:,1),'-r');
plotGauss2D(MEAN(:,:,2),COV(:,:,2),'-g');
plotGauss2D(MEAN(:,:,3),COV(:,:,3),'-b');

%% 4(d)

bc = gaussBayesClassify( Xtr, Ytr );
figure;
plotClassify2D(bc, Xtr, Ytr);

%% 4(e)
%old =cd('')
Yhat = predict( bc, Xtr );
errTrain = mean(Yhat~=Ytr);
Yhat_test=predict(bc,Xte);
errTest=mean(Yhat_test~=Yte);

%% 4(f)
iris=load('data/iris.txt'); 

y=iris(:,end); 
x=iris(:,1:end-1);
[x, y] = shuffleData(x,y);
[Xtr,Xte ,Ytr, Yte] = splitData(x,y, .60);

bc = gaussBayesClassify( Xtr, Ytr );
Yhat = predict( bc, Xtr );
errTrain_all = mean(Yhat~=Ytr);
Yhat_test=predict(bc,Xte);
errTest_all=mean(Yhat_test~=Yte);