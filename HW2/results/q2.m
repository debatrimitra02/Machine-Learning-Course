clear all;

%% 2
DATA=load('data/curve80.txt');                    % read data from text
y=DATA(:,end); 
x=DATA(:,1);
[Xtr,Xte ,Ytr, Yte] = splitData(x,y, .75);

% lr = linearRegress( Xtr, Ytr );
% lr1 = linearRegress( Xte, Yte );
% xs = [0:.05:10]';
% ys = predict( lr, xs );
% figure, 
% plot(xs,ys);
% hold on;
% scatter(Xtr,Ytr);
% 
% 
% ytr_Hat=predict(lr,Xtr);
% yte_Hat=predict(lr1,Xte);
% Mse=[(transpose(yte_Hat-Yte)*(yte_Hat-Yte))/(0.25*length(x))];
% Mse_train=[(transpose(ytr_Hat-Ytr)*(ytr_Hat-Ytr))/(0.75*length(x))];

j=1;
for d=[1, 3, 5, 7, 10, 18]
    
    nFolds = 5;
    for iFold = 1:nFolds,
    
        [Xti,Xvi,Yti,Yvi] = crossValidate(Xtr,Ytr,nFolds,iFold); % take ith data block as validation
        XtrP = fpoly(Xti,d, false);
        [XtrP, M,S] = rescale(XtrP);
        learner = linearRegress(XtrP,Yti);
        
        xsP= fpoly(Xvi,d,false);
        [xsP] = rescale(xsP,M,S);
        ysP=predict(learner,xsP);
    % TODO: train on Xti, Yti , the data for this fold
    J(iFold) = [(transpose(ysP-Yvi)*(ysP-Yvi))/(length(Xvi))];
    % TODO: now compute the MSE on Xvi, Yvi and save it
    end;
    MSE(j)=mean(J);
    j=j+1;
end
figure; plot([1 3 5 7 10 18],log(MSE),'r-')
% hold on;
% plot([1 3 5 7 10 18],log(M_se1),'b-');
