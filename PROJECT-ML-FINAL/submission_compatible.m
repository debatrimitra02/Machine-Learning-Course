function  submission_compatible( filename,prediction )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for i=1:40000
one(i,1)=i;
end

prediction=cat(2,one,prediction);
%prediction=cat(1,first,prediction);
csvwrite(filename,prediction);

end

