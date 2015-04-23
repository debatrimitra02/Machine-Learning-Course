clear all;
%old=cd('C:\Users\DEBATRI\Desktop\HW1\data');  % cd to the data folder
DATA=load('data/iris.txt');                    % read data from text  
%cd(old);                                      % cd back to old directory

x=DATA(:,1:end-1);                            % features  
y=DATA(:,end);                                % classes

whos;                                         % show current variables in memory and sizes

no_of_features = size(x,2);                   % get no of features i.e. equal to num of columns  
no_of_data = size(x,1);                       % get no of data points i.e. equal to the no of rows 

%size(x)
MEAN =zeros(4,1);
VAR=zeros(4,1);

for i=1:no_of_features
    subplot(2,2,i);
    hist(x(:,i));
    MEAN(i,1)=mean(x(:,i));
    VAR(i,1)=var(x(:,i));
end
STD=VAR.^(1/2);
%NORMAL=zeros(size(x));
ONES=ones(no_of_data,1);


for i=1:no_of_features
    x(:,i)=(x(:,i)-ONES.*MEAN(i,1));
    x(:,i)=x(:,i)./STD(i,1);
end
%STD=zeros(4,1);

figure;

subplot(3,1,1);
gscatter(x(:,1),x(:,2),y(:,1),'bgr','xo+');
xlabel('feature1');
ylabel('feature2');
subplot(3,1,2);
gscatter(x(:,1),x(:,3),y(:,1),'bgr','xo+');
xlabel('feature1');
ylabel('feature3');
subplot(3,1,3)
gscatter(x(:,1),x(:,4),y(:,1),'bgr','xo+');
xlabel('feature1');
ylabel('feature4');
%title('Scatter Plot')