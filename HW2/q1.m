clear all;

%% 1(a)
DATA=load('data/curve80.txt');                    % read data from text
y=DATA(:,end); 
x=DATA(:,1);
[Xtr,Xte ,Ytr, Yte] = splitData(x,y, .75);


%% 1(b)

lr = linearRegress( Xtr, Ytr );
lr1 = linearRegress( Xte, Yte );
xs = [0:.05:10]';
ys = predict( lr, xs );
figure, 
plot(xs,ys);
hold on;
scatter(Xtr,Ytr);


ytr_Hat=predict(lr,Xtr);
yte_Hat=predict(lr1,Xte);
Mse=[(transpose(yte_Hat-Yte)*(yte_Hat-Yte))/(0.25*length(x))];
Mse_train=[(transpose(ytr_Hat-Ytr)*(ytr_Hat-Ytr))/(0.75*length(x))];

%% 1(c)
j=0;
for d=[1, 3, 5, 7, 10, 18]
XtrP = fpoly(Xtr,d, false); % create polynomial features up to given degree,

[XtrP, M,S] = rescale(XtrP); % often a good idea to scale the features
lr = linearRegress( XtrP, Ytr ); % create and train model
xsP= fpoly(xs,d,false);
[xsP] = rescale(xsP,M,S);
ysP=predict(lr,xsP);
ytrP=predict(lr,XtrP);
XteP = fpoly(Xte,d, false); % create polynomial features up to given degree,

[XteP] = rescale(XteP,M,S); % often a good idea to scale the features
yteP=predict(lr,XteP);

subplot(3,2,j+1)
scatter(Xtr,Ytr);
hold on;
plot(xs,ysP);
j=j+1;
M_se(j)=[(transpose(abs(yteP-Yte))*(abs(yteP-Yte)))/(0.25*length(x))];
M_se1(j)=[(transpose(abs(ytrP-Ytr))*(abs(ytrP-Ytr)))/(0.75*length(x))];
end
figure; plot(([1 3 5 7 10 18]),log(M_se),'r-')
hold on;
plot(([1 3 5 7 10 18]),log(M_se1),'b-');

