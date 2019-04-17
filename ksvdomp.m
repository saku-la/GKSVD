% function [ytest,res,sim,acc]=ksvdomp(dim)
function [ytest,res,sim,acc,g]=ksvdomp(dim)
%ytest ����
%res Ԥ�����
%r ���
%C ׼ȷ��
tic
%------------ read the data --------------
data=importdata('GEMS/DLBCL.mat');%ÿһ�б�ʾһ����������һ��������
% [data]=importdata('data/yale.mat');
data=sortrows(data,1);
[num,width]=size(data);
classnum=max(data(:,1));
classcode=0:classnum;
A=mapminmax(data(:,2:width),0,1);% ���а����ݹ�һ��[-1,1]
% A=data(:,2:width);%���ݼ�ֱ�Ӹ�ֵ
%% PCA+SVD
[U]=PCA(A);%�����Ѿ���PCA��SVD����
U=U(:,1:dim);
A=A*U;

ytest=data(:,1);
%% run k-svd training %%
dictsize =dim+2;
params.data = A';
params.dictsize = dictsize;
params.iternum = 10;
params.memusage = 'high';
tttt=1;
params.Tdata=dim/4;

cd '.\ksvdbox13';

[D,g,err] = ksvd(params,'');
g=full(g);
cd '..';
%% ѡ���������

xapp=D;


for j=1:num
    y=A(j,:);
    yapp=data(:,1);
    yapp(j,:)=[];

    hat_x=cs_omp(y',xapp,dictsize); 
    for i=0:classnum
        pos=find(yapp==i);
        data_rec=mean(g(:,pos),2);
        sim(j,i+1)=abs(hat_x*data_rec)/(norm(hat_x)*norm(data_rec));

        
    end
    [val_r,pos_r]=max(sim(j,:));
    res(j,tttt)=pos_r-1;


end
% res=res';
[C,metric]=ConfusionMatrix(res(:,tttt),ytest,classcode);
acc(tttt)=metric.accuracy;
% end
toc



%************************************************************************%
function hat_x=cs_omp(y,T_Mat,m)
% y=T_Mat*x, T_Mat is n-by-m
% y - measurements
% T_Mat - combination of random matrix and sparse representation basis
% m - size of the original signal
% the sparsity is length(y)/4

n=length(y);
[h,w]=size(T_Mat);
s=(m-2)/4;
% if h>w
%     s=floor(w/4);                                     %  ����ֵά��
% else
%     s=floor(h/4);
% end
hat_x=zeros(1,m);                                 %  ���ع�������(�任��)����                     
Aug_t=[];                                         %  ��������(��ʼֵΪ�վ���)
r_n=y;                                            %  �в�ֵ 
for times=1:s;                                  %  ��������(ϡ����ǲ�����1/4)

    product=abs(T_Mat'*r_n);
    [val,pos]=max(product);                       %  ���ͶӰϵ����Ӧ��λ��
    Aug_t=[Aug_t,T_Mat(:,pos)];                   %  ��������
    T_Mat(:,pos)=zeros(n,1);                      %  ѡ�е������㣨ʵ����Ӧ��ȥ����Ϊ�˼򵥽������㣩
   %% Ϊ�˱������������־���ӽ�����ֵ���������svd����ֱ������
    T=Aug_t'*Aug_t;
    [tr,tc]=size(T);
    [UU,S,VV] = svd(T);
    summ=sum(diag(S));
    S1=S;
    for i=1:tr
        S1(i,i)=1/S(i,i);
        if sum(diag(S(1:i,1:i)))/summ>0.9999
            S1(i+1:tr,i+1:tr)=0;
            break;
        end
    end 
    aug_x=VV*S1*UU'*Aug_t'*y;
    
%%
%     aug_x=(Aug_t'*Aug_t)^(-1)*Aug_t'*y;           %  ��С����,ʹ�в���С
    r_n=y-Aug_t*aug_x;                            %  �в�
    pos_array(times)=pos;                         %  ��¼���ͶӰϵ����λ��
    
end
hat_x(pos_array)=aug_x;                           %  �ع������� 