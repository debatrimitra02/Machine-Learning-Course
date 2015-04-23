clear all;
iris=load('data/iris.txt');
X = iris(:,1:2); Y=iris(:,end);
[X,Y] = shuffleData(X,Y);
X = rescale(X);
XA = X(Y<2,:); YA=Y(Y<2);
XB = X(Y>0,:); YB=Y(Y>0);

%% problem-a %%
figure;
gscatter(XA(:,1),XA(:,2),YA);
figure;
gscatter(XB(:,1),XB(:,2),YB);

%% Problem-b %%
learner=logisticClassify2();
learner=setClasses(learner, unique(YA));
wts = [0.5 1 -0.25];
learner=setWeights(learner, wts);

figure;
plot2DLinear(learner,XA,YA)
figure;
plot2DLinear(learner,XB,YB)

figure;
plotClassify2D(learner,XA,YA)
figure;
plotClassify2D(learner,XB,YB)

%figure;
%plot2DLinear(learner,XA,YA)

%% problem -c %%
