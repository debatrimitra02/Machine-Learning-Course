function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;

  u=unique (Y);
  class0 = find(Y==u(1));
  class1= find(Y==u(2));
  Xplt = linspace(min(X(:,1)),max(X(:,1)),200);
  plot(X(class0,1),X(class0,2),'ro',...
  X(class1,1),X(class1,2),'gx',...
  Xplt,-obj.wts(1)/obj.wts(3) - obj.wts(2)/obj.wts(3).*Xplt,'b-');
% TODO: Plot each class in a different color
% along with the linear decision boundary of the predictor
drawnow;
% ensures plot is updated immediately

  %%% TODO: Fill in the rest of this function...  
