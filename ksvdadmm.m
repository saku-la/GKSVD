function [ytest,res,sim,acc]=ksvdadmm(dim)
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
%% guass
% U=randn(width-1,dim);
% A=A*U;
%% relief-F
%D为输入的训练集合,输入集合去掉身份信息项目;m为循环选择操作次数，k为最近邻样本个数
% class_info=tabulate(data(:,1));
% K=min(class_info(:,2))-1;
% if K>=ceil(num/(classnum+4))&&K<ceil(num/(classnum+2));
%     K=ceil(K/2);
% else if K>=ceil(num/(classnum+2))
%         K=ceil(K/4);
%     end
% end
% W = F_ReliefF (A,data(:,1),20,K);
% [~,pos]=sort(abs(W));
% A=A(:,pos(width-dim:width-1));

% [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(A,data(:,1),floor(num*0.8), classcode);

ytest=data(:,1);

%% run k-svd training %%
dictsize = dim+2;
params.data = A';
% params.Tdata = ceil(num/20);
params.dictsize = dictsize;
params.iternum = 10;
params.memusage = 'high';
tttt=1;
params.Tdata=dim/4;
% for tttt=1:dim/2;
% params.Tdata=tttt;

cd '.\ksvdbox13';
[D,g,err] = ksvd(params,'');
g=full(g);
cd '..';

z=zeros(dictsize ,1);
v=zeros(dictsize ,1);

%% 选择参照样本
% for i=0:classnum
%     pos=find(ytest==i);
% %     referid(1,i+1)=min(pos);
%     xi=mean(A(min(pos):max(pos),:));
%     distxi = pdist2(A(min(pos):max(pos),:),xi, 'euclidean');
%     [~,post]=min(distxi);
%     referid(1,i+1)=min(pos)-1+post;
% end
% refer=A(referid,:);
xapp=D;
% for i=0:classnum
%     lambda_max = norm(xapp'*refer(i+1,:)', 'inf');
%     lambda = 0.1*lambda_max;
%     refer_x(i+1,:)=admm(xapp,refer(i+1,:)',lambda,1.0,z,v,dictsize); 
% end

for j=1:num
    y=A(j,:);
%     xapp=A;
%     xapp(j,:)=[];
    yapp=data(:,1);
    yapp(j,:)=[];
    lambda_max = norm(xapp'*y', 'inf');
    lambda = 0.1*lambda_max;
    [hat_x,z]= admm(xapp,y',lambda,1.0,z,v,dictsize);
    for i=0:classnum
%                 refer_x=cs_omp(refer(i+1,:)',xapp,dictsize);
%                 sim(j,i+1)=norm(hat_x-refer_x(i+1,:));
%                 sim(j,i+1)=hat_x*refer_x(i+1,:)'/(norm(hat_x)*norm(refer_x(i+1,:)));
%                 sim(j,i+1)=hat_x*refer_x(i+1,:)';
        pos=find(yapp==i);
        data_rec=mean(g(:,pos),2);%xapp(:,pos)*hat_x(pos)';
%         sim(j,i+1)=norm(hat_x-data_rec); 
        sim(j,i+1)=abs(hat_x'*data_rec)/(norm(hat_x)*norm(data_rec));
    end
    [val_r,pos_r]=max(sim(j,:));
%     [val_r,pos_r]=min(sim(j,:));
    res(j,tttt)=pos_r-1;

%     for i=0:classnum
%         pos=find(yapp==i);
%         xapp_i=xapp(min(pos):max(pos),:);
%         column_rec=cs_omp(y',xapp_i',max(pos)-min(pos)+1);
%         P=column_rec;           % sparse representation
%         data_rec=P*xapp_i;          % inverse transform
%         r(j,i+1)=sum((y-data_rec).^2);
%     end
%     [val_r,pos_r]=min(r(j,:));
%     res(j)=pos_r-1;
end
% res=res';
[C,metric]=ConfusionMatrix(res(:,tttt),ytest,classcode);
acc(tttt)=metric.accuracy;
% end
toc
end
 %% admm函数   
function [x,z]=admm(A,y,lambda,rho,z,v,n)
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
MAX_ITER = 16;
[row]=size(A);
I1=eye(row(:,2));
for k = 1:MAX_ITER
    zold = z;
    x = (A'*A+rho*I1)^-1*(A'*y+rho*(z-v));
    z = shrinkage(x + v, lambda/rho);
    v = v + (x - z);
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*v);
    if (history.r_norm(k) < history.eps_pri(k) && ...
      history.s_norm(k) < history.eps_dual(k))
        break;
    end
end
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

