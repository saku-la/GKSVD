% function [ytest,res,sim,acc]=ksvdomp(dim)
function [ytest,res,sim,acc,g]=ksvdomp(dim)
%ytest 类别号
%res 预测类别
%r 误差
%C 准确率
tic
%------------ read the data --------------
data=importdata('GEMS/DLBCL.mat');%每一行表示一个样本，第一列是类别号
% [data]=importdata('data/yale.mat');
data=sortrows(data,1);
[num,width]=size(data);
classnum=max(data(:,1));
classcode=0:classnum;
A=mapminmax(data(:,2:width),0,1);% 按行把数据归一化[-1,1]
% A=data(:,2:width);%数据集直接赋值
%% PCA+SVD
[U]=PCA(A);%这里已经将PCA与SVD相结合
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
%% 选择参照样本

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
%     s=floor(w/4);                                     %  测量值维数
% else
%     s=floor(h/4);
% end
hat_x=zeros(1,m);                                 %  待重构的谱域(变换域)向量                     
Aug_t=[];                                         %  增量矩阵(初始值为空矩阵)
r_n=y;                                            %  残差值 
for times=1:s;                                  %  迭代次数(稀疏度是测量的1/4)

    product=abs(T_Mat'*r_n);
    [val,pos]=max(product);                       %  最大投影系数对应的位置
    Aug_t=[Aug_t,T_Mat(:,pos)];                   %  矩阵扩充
    T_Mat(:,pos)=zeros(n,1);                      %  选中的列置零（实质上应该去掉，为了简单将其置零）
   %% 为了避免矩阵求逆出现矩阵接近奇异值的情况，用svd代替直接求逆
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
%     aug_x=(Aug_t'*Aug_t)^(-1)*Aug_t'*y;           %  最小二乘,使残差最小
    r_n=y-Aug_t*aug_x;                            %  残差
    pos_array(times)=pos;                         %  纪录最大投影系数的位置
    
end
hat_x(pos_array)=aug_x;                           %  重构的向量 