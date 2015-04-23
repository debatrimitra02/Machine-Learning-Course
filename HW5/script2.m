clear all
[vocab] = textread('vocab.txt','%s');
[did,wid,cnt] = textread('docword.txt','%d%d%d','headerlines',3);
X=sparse(did,wid,cnt);
D=max(did);
W=max(wid);
N=sum(cnt);
Xn= X./ repmat(sum(X,2),[1,W]) ;
[z,c,sumd] = Kmeans(Xn,20,'k++'); 
display(sumd);

ssd = inf;
for it=1:5,
[Zi,mui,ssdi] = Kmeans(Xn,20,'k++'); ssdi, % compute 20 clusters
if (ssd > ssdi) Z=Zi; mu=mui; ssd=ssdi; end;
end;


h= hist(Z,1:20);
hist(Z,1:40)

for i=1:20,
[sorted,order] = sort( mu(i,:), 2, 'descend');
fprintf(' %d: ',i); fprintf('%s ',vocab{order(1:10)}); fprintf('\n');
end
Z(1), Z(15), Z(30)


lst = find(Z==Z(1)); lst=lst(1:min(length(lst),12));
for i=lst',
fname = sprintf('example1/20000101.%04d.txt',i);
txt = textread(fname,'%s',2,'whitespace','\r\n'); fprintf('%s\n',txt{:}); fprintf('\n');
end;
