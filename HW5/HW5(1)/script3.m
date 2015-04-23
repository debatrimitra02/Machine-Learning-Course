clear all
X = load('data/faces.txt');
% load face dataset
%img = reshape(X(i,:),[24 24]); % convert vectorized datum to 24x24 image patch
%imagesc(img); axis square; colormap gray;

mu = mean(X);
X0 = X - repmat(mu,[size(X,1),1]);
[U S V] = svds(X0,25);
W = U*S;
for k=1:10,
X_predict = W(:,1:k)*V(:,1:k)';
E(k) = mean(mean( (X0 - X_predict).^2 ));
end;
figure; plot(1:10, E);

for k=3,
alpha = 2*median(abs(W(:,k)));
im1 = reshape(mu + alpha*V(:,k)', [24 24]);
im2 = reshape(mu - alpha*V(:,k)', [24 24]);
figure;
subplot(2,1,1);imagesc(im1); colormap gray;
subplot(2,1,2);imagesc(im2); colormap gray;
end;

idx = [1:30];
% pick some data
figure; hold on; axis ij; colormap(gray);
range = max(W(idx,1:2)) - min(W(idx,1:2)); % find range of coordinates to be plotted
scale = [200 200]./range;
for i=idx, imagesc(W(i,1)*scale(1),W(i,2)*scale(2), reshape(X(i,:),24,24)); end;

