clear all;
iris=load('data/iris.txt');
X = iris(:,1:2); 
Y=iris(:,end); 
XA = X(Y<2,:); 
YA=Y(Y<2);
YA(find(YA==0)) = -1;
gscatter(XA(:,1),XA(:,2),YA);

m = size(XA,1) ;
n = size(XA,2) ;

H=eye(n+1);
H(n+1,n+1)=0 ;

f=zeros(n+1,1);
Z = [XA ones(m,1)];
A=-diag(YA)*Z ;
c=-1*ones(m,1) ;
w=quadprog(H,f,A,c)

learner=logisticClassify(XA,YA);
learner=setWeights(learner, w');
learner=setClasses(learner, unique(YA));
%pc=setWeights(pc,w);
plotClassify2D(learner,XA,YA);

