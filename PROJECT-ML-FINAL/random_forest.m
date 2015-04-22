clear all;
X= load('kaggle.X1.train.txt');
Y= load('kaggle.Y.train.txt');
[Xt Xv Yt Yv] = splitData(X,Y,.8);
Xe = load('kaggle.X1.test.txt');

% rf = cell(1,25);
% YtHat = zeros(size(Yt,1),25);
% % we'll just make the predictions at
% YvHat = zeros(size(Yv,1),25);
% 
% for l=1:25,
% % (This can take a while!)
% [Xi Yi] = bootstrapData(Xt,Yt,size(Xt,1)); % bootstrap sample for this learner
% rf{l} = treeRegress(Xi,Yi, 'maxDepth',15, 'nFeatures',60); % train & save learner
% YtHat(:,l) = predict(rf{l},Xt);
% % predict on training
% YvHat(:,l) = predict(rf{l},Xv);
% 
% end;

nEnsemble = 30;
Ye = zeros(size(Xe,1),1);
for l=1:nEnsemble,
% (This can take a while!)
[Xi Yi] = bootstrapData(X,Y,size(X,1));
% bootstrap sample for this learner
rf{l} = treeRegress(Xi,Yi, 'maxDepth',20, 'nFeatures',60); % train next tree
Ye = Ye + predict( rf{l}, Xe);
% build and predict
end;
Ye = Ye / nEnsemble;

