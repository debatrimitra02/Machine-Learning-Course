% function Yte = predict(obj,Xte)
% % Yhat = predict(obj, X)  : make predictions on test data X
% 
% % (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% 
% % (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );
% Yte = sign( obj.wts(1) + Xte*obj.wts(2:end)');   
% c=getClasses(obj); 
% Yte = obj.classes;
%  %Yte = setClasses( Yte/2 + 1.5 );

function Yte = predict(obj,Xte)
z = ( obj.wts(1) + Xte*obj.wts(2:end)');
sig = (1+exp(-z)).^(-1);
for i=1:size(sig)
if(sig(i) >0.5)
    Yte(i) = 1;
else
    Yte(i) = 0;
 
end
end
end
