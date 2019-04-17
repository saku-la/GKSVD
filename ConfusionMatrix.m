function   [C,metric]=ConfusionMatrix(ypred,ytrue,classcode)
%  [C,metric]=ConfusionMatrix(ypred,ytrue,classcode)
%
%  ypred and ytrue -1, 1
%  if classcode = 1 -1
%
%  TP FN
%  FP TN
% load('ypred');
% load('ytrue');
% ytrue=ytest;
% classcode=[1 2 3 4];
C=zeros(2);
N=length(classcode);
for i=1:N
    for j=1:N
        C(i,j)=  length(  find(ypred==classcode(j) & ytrue == classcode(i))); 
    end;
end;
% 各类数据集个数
% nbpos=0;
% nbneg=0;
% for i=1:N/2
% nbpos=sum(ytrue==i)+nbpos;
% nbneg=sum(ytrue==N/2+i)+nbneg;
% end
% c=nbneg/nbpos;
% metric.detection=C(1,1)/sum(C(1,:));
metric.accuracy=sum(diag(C))/sum(sum(C));%正确率，accuracy = （TP+TN）/(P+N)
for i=1:N
    metric.sensitivity(1,i)=sum(diag(C(i,i)))/sum(C(i,:));
end
% metric.TPRsensitivity=sum(diag(C(1:N/2,1:N/2)))/nbpos;% 灵敏度sensitive = TP/P，表示的是所有正例中被分对的比例
% metric.FNR=sum(sum(C(1:N/2,N/2+1:N)))/nbpos;%假负率
% metric.FPR=sum(sum(C(N/2+1:N,1:N/2)))/nbpos;%假正率
% metric.TNRspecificity=sum(diag(C(N/2+1:N,N/2+1:N)))/nbneg;% 特指度specificity = TN/N，表示的是所有负例中被分对的比例
% if sum(C(:,1))~=0
% metric.precision=sum(sum(C(1:N/2,1:N/2)))/sum(sum(C(1:N,1:N/2)));% 精度是精确性的度量，表示被分为正例的示例中实际为正例的比例，precision=TP/（TP+FP）；
% else
%     metric.precision=NaN;
% end;
% metric.recall=sum(sum(C(1:N/2,1:N/2)))/sum(sum(C(1:N/2,1:N)));% 召回率是覆盖面的度量，度量有多个正例被分为正例，recall=sensitivity

% metric
% C

