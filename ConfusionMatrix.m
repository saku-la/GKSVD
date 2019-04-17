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
% �������ݼ�����
% nbpos=0;
% nbneg=0;
% for i=1:N/2
% nbpos=sum(ytrue==i)+nbpos;
% nbneg=sum(ytrue==N/2+i)+nbneg;
% end
% c=nbneg/nbpos;
% metric.detection=C(1,1)/sum(C(1,:));
metric.accuracy=sum(diag(C))/sum(sum(C));%��ȷ�ʣ�accuracy = ��TP+TN��/(P+N)
for i=1:N
    metric.sensitivity(1,i)=sum(diag(C(i,i)))/sum(C(i,:));
end
% metric.TPRsensitivity=sum(diag(C(1:N/2,1:N/2)))/nbpos;% ������sensitive = TP/P����ʾ�������������б��ֶԵı���
% metric.FNR=sum(sum(C(1:N/2,N/2+1:N)))/nbpos;%�ٸ���
% metric.FPR=sum(sum(C(N/2+1:N,1:N/2)))/nbpos;%������
% metric.TNRspecificity=sum(diag(C(N/2+1:N,N/2+1:N)))/nbneg;% ��ָ��specificity = TN/N����ʾ�������и����б��ֶԵı���
% if sum(C(:,1))~=0
% metric.precision=sum(sum(C(1:N/2,1:N/2)))/sum(sum(C(1:N,1:N/2)));% �����Ǿ�ȷ�ԵĶ�������ʾ����Ϊ������ʾ����ʵ��Ϊ�����ı�����precision=TP/��TP+FP����
% else
%     metric.precision=NaN;
% end;
% metric.recall=sum(sum(C(1:N/2,1:N/2)))/sum(sum(C(1:N/2,1:N)));% �ٻ����Ǹ�����Ķ����������ж����������Ϊ������recall=sensitivity

% metric
% C

