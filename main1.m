% Computational Finance CW2
% Question 1
load options.mat
%% Q1
optionNum=1;
strikePrices=[2925 3025 3125 3225 3325];
L=length(stockPrice);% data length
Lwin=fix(L/4);% window length
Lrest=L-Lwin;% rest length
Tt=ones(L,1);
for i=1:L
    Tt(i)=(L-i)/252;
end
X=[stockPrice(1:L)/strikePrices(optionNum) Tt];

GMModel = fitgmdist(X,4);
% figure(1),clf,
% scatter(X(:,1),X(:,2),'r.');
% hold on
% ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
% title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
% hold off
% grid on

m1=GMModel.mu(1,:);
m2=GMModel.mu(2,:);
m3=GMModel.mu(3,:);
m4=GMModel.mu(4,:);
C1=GMModel.Sigma(:,:,1);
C2=GMModel.Sigma(:,:,2);
C3=GMModel.Sigma(:,:,3);
C4=GMModel.Sigma(:,:,4);

designMat=ones(L,7);% design matrix
designMat(:,5)=X(:,1);
designMat(:,6)=X(:,2);

for i=1:L
    designMat(i,1)=sqrt((X(i)-m1)*C1*(X(i)-m1)');
    designMat(i,2)=sqrt((X(i)-m2)*C2*(X(i)-m2)');
    designMat(i,3)=sqrt((X(i)-m3)*C3*(X(i)-m3)');
    designMat(i,4)=sqrt((X(i)-m4)*C4*(X(i)-m4)');    
end

% true normalized call option price
CXtrue=optionCPrice(1:L,optionNum)/strikePrices(optionNum);

% train
cvx_begin quiet
variable w(7)
minimize( norm(designMat*w-CXtrue) )
cvx_end
%%
x1=linspace(0,2,100);
x2=linspace(0,2,100);

% syms x1 x2 CX

CX=w(1)*sqrt(([x1,x2]-m1)*C1*([x1,x2]-m1)')...
    +w(2)*sqrt(([x1,x2]-m2)*C2*([x1,x2]-m2)')...
    +w(3)*sqrt(([x1,x2]-m3)*C3*([x1,x2]-m3)')...
    +w(4)*sqrt(([x1,x2]-m4)*C4*([x1,x2]-m4)')...
    +[x1,x2]*[w(5);w(6)]+w(7);
%%
ezplot3(CX);




