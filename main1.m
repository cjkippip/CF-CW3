% Computational Finance CW2
% Question 1
load options.mat
%% Q1
optionNum=2;% number of strike price
strikePrices=[2925 3025 3125 3225 3325];
L=length(stockPrice);% data length 222
Lwin=fix(L/4);% window length 55
Lrest=L-Lwin;% rest length 167
% LTrain=fix(Lrest*13/14)+1; 
% LTest=Lrest-LTrain; 
LTrain=120; % train data length
LTest=47; % test data length 
Tt=ones(L,1);
for i=1:L
    Tt(i)=(L-i)/252;
end

% data: X=[S/X T-t]'
X=[stockPrice(56:L)/strikePrices(optionNum) Tt(56:L,1)];
XTrain=X(1:LTrain,:); % train data
Xtest=X(LTrain+1:Lrest,:); % test data

% true normalized call option price
CXtrue=optionCPrice(56:L,optionNum)./strikePrices(optionNum);
CXtrueTrain=CXtrue(1:LTrain,:);
CXtrueTest=CXtrue(LTrain+1:Lrest,:);

%% GMModel generate 4 means and covariances
GMModel = fitgmdist(XTrain,4,'RegularizationValue',0.0003);
% GMModel = fitgmdist(XTrain,4);
figure(1),clf,
scatter(XTrain(:,1),XTrain(:,2),'ro');
hold on
ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
axis([0.8 1.3 -0.1 0.8]);
xlabel('S/X','FontSize',13,'FontWeight','bold');
ylabel('T-t','FontSize',13,'FontWeight','bold');
hold off
%% disign matrix
m1=GMModel.mu(1,:);
m2=GMModel.mu(2,:);
m3=GMModel.mu(3,:);
m4=GMModel.mu(4,:);
C1=GMModel.Sigma(:,:,1);
C2=GMModel.Sigma(:,:,2);
C3=GMModel.Sigma(:,:,3);
C4=GMModel.Sigma(:,:,4);

designMat=ones(LTrain,7);% design matrix
designMat(:,5)=XTrain(:,1);
designMat(:,6)=XTrain(:,2);

for i=1:LTrain
    designMat(i,1)=sqrt((XTrain(i,:)-m1)*C1*(XTrain(i,:)-m1)');
    designMat(i,2)=sqrt((XTrain(i,:)-m2)*C2*(XTrain(i,:)-m2)');
    designMat(i,3)=sqrt((XTrain(i,:)-m3)*C3*(XTrain(i,:)-m3)');
    designMat(i,4)=sqrt((XTrain(i,:)-m4)*C4*(XTrain(i,:)-m4)');    
end

%% train using cvx
cvx_begin quiet
variable w(7)
minimize( norm(designMat*w-CXtrueTrain) )
cvx_end
%%
CXNum=30;
x=linspace(0.7,1.5,CXNum);
y=linspace(0.7,0,CXNum);

CX=ones(CXNum,CXNum);
for i=1:CXNum
    for j=1:CXNum
        CX(j,i)=w(1)*sqrt(([x(i),y(j)]-m1)*C1*([x(i),y(j)]-m1)')...
            +w(2)*sqrt(([x(i),y(j)]-m2)*C2*([x(i),y(j)]-m2)')...
            +w(3)*sqrt(([x(i),y(j)]-m3)*C3*([x(i),y(j)]-m3)')...
            +w(4)*sqrt(([x(i),y(j)]-m4)*C4*([x(i),y(j)]-m4)')...
            +[x(i),y(j)]*[w(5);w(6)]+w(7);
    end
end

figure(2),clf,
mesh(x,y,CX);
xlabel('S/X','FontSize',13,'FontWeight','bold')
ylabel('T-t','FontSize',13,'FontWeight','bold')
zlabel('C/X','FontSize',13,'FontWeight','bold')
% axis([0.8 1.5 0 0.6 0 0.4]);
grid on
grid minor
%%
CXpred=ones(Lrest,1);
for i=1:Lrest
    CXpred(i)=w(1)*sqrt((X(i,:)-m1)*C1*(X(i,:)-m1)')...
        +w(2)*sqrt((X(i,:)-m2)*C2*(X(i,:)-m2)')...
        +w(3)*sqrt((X(i,:)-m3)*C3*(X(i,:)-m3)')...
        +w(4)*sqrt((X(i,:)-m4)*C4*(X(i,:)-m4)')...
        +X(i,:)*[w(5);w(6)]+w(7);
end

figure(3),clf,
xx1=56:222;
plot(xx1,CXpred,'b','LineWidth',1.5);
xlabel('Date','FontSize',13,'FontWeight','bold')
ylabel('C/X','FontSize',13,'FontWeight','bold')
hold on
plot(xx1,CXtrue,'r','LineWidth',1.5);
axis([-inf,inf,-inf,inf]);
legend({'predicted','real'},'Location','northwest',...
    'FontSize',13,'FontWeight','bold');
plot([205,205],[0,0.12],'k','LineWidth',2);
grid on
grid minor
hold off










