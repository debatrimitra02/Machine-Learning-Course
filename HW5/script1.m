clear all;
data = load('data/iris.txt'); 
X=data(:,1:2);
figure;
scatter(X(:,1),X(:,2));

%[idx,C] = kmeans(X,5,'Distance','cityblock','Replicates',5,'Options',opts);
[assign, clusters, sumd] = Kmeans(X,5,'k++'); % 5 clusters
figure; plotClassify2D([],X,assign); 
sumd
[assign, clusters, sumd] = Kmeans(X,20,'k++'); % 5 clusters
figure; plotClassify2D([],X,assign); 
sumd

cluster = agglomCluster(X,5,'min'); % single linkage 
figure; plotClassify2D([],X,cluster);
cluster = agglomCluster(X,5,'max'); % complete linkage
figure; plotClassify2D([],X,cluster);

cluster = agglomCluster(X,20,'min'); % single linkage
figure; plotClassify2D([],X,cluster);

cluster = agglomCluster(X,20,'max'); % complete linkage
figure; plotClassify2D([],X,cluster);

[z,T,soft,ll]= emCluster(X,5,'k++'); 
display(ll);

[z,T,soft,ll] = emCluster(X,20,'k++');
display(ll);



