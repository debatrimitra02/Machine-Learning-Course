clear all;

iris=load('data/iris.txt'); 

y=iris(:,end); 
x=iris(:,1:2);
[x, y] = shuffleData(x,y);
[Xtr,Xte ,Ytr, Yte] = splitData(x,y, .75);


%% Problem- 2(a)
j=1;
for K=[10, 50, 100, 200]
    
knn = knnClassify( Xtr, Ytr, K );
YteHat = predict( knn, Xte );

subplot(2,2,j)
plotClassify2D( knn, Xtr, Ytr );
j=j+1;
end

%% Problem- 2(b)
%K=[1,2,5,10,50,100,200];
i=1;
for K=[1,2,5,10,50,100,200];    
model = knnClassify( Xtr, Ytr, K );
Yhat = predict( model, Xtr );

%model = knnClassify( Xtr, Ytr, K );
Yhat_test = predict( model, Xte );

errTrain(i) = mean(Yhat~=Ytr);
errTest(i) =  mean(Yhat_test~=Yte);

i=i+1;
end

K=[1,2,5,10,50,100,200];
semilogx(K,errTrain,'r');
hold on;
semilogx(K,errTest,'g');

