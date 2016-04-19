% Computational Finance CW2
% Question 1
load options.mat
%%
optionNum=4;% number of strike price
strikePrices=[2925 3025 3125 3225 3325];

L=length(stockPrice);% data length 222
Lwin=fix(L/4);% window length 55
Lrest=L-Lwin;% rest length 167
% LTrain=fix(Lrest*13/14)+1; 
% LTest=Lrest-LTrain; 
LTrain=700; % train data length
LTest=135; % test data length 
LAll=835; % all data length

% interval_Tr=(optionNum-1)*Lrest+1:(optionNum-1)*Lrest+LTrain;
% interval_Ts=(optionNum-1)*Lrest+LTrain+1:(optionNum-1)*Lrest+Lrest;
% interval_All=(optionNum-1)*Lrest+1:(optionNum-1)*Lrest+Lrest;

interval_Tr=1:700;
interval_Ts=701:835;
interval_All=1:835;

Tt=ones(L,1);
for i=1:L
    Tt(i)=(L-i)/252;
end

% data: X=[S/X T-t]
X=[stockPrice(56:L)/strikePrices(1) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(2) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(3) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(4) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(5) Tt(56:L,1)];

XTrain=X(interval_Tr,:); % train data
XTest=X(interval_Ts,:); % test data
XAll=X(interval_All,:); % all data

% normalized call option price from BS formula
CXTrue=[BSOptionCPrices(:,1)./strikePrices(1);...
    BSOptionCPrices(:,2)./strikePrices(2);...
    BSOptionCPrices(:,3)./strikePrices(3);...
    BSOptionCPrices(:,4)./strikePrices(4);...
    BSOptionCPrices(:,5)./strikePrices(5)];

CXtrueTrain=CXTrue(interval_Tr,:); % train tag
CXtrueTest=CXTrue(interval_Ts,:); % test tag
CXtrueAll=CXTrue(interval_All,:); % all tag
%% GMModel generate 4 means and covariances
GMModel = fitgmdist(XTrain,4,'RegularizationValue',0.0003);
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
%% draw surface
CXNum=30;
x=linspace(0.8,1.2,CXNum);
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
plot3(X(1:167,1),X(1:167,2),CXTrue(1:167),'o',...
    'MarkerSize',8,'MarkerFaceColor','b');
hold on
plot3(X(168:334,1),X(168:334,2),CXTrue(168:334),'o',...
    'MarkerSize',8,'MarkerFaceColor','r');
plot3(X(335:501,1),X(335:501,2),CXTrue(335:501),'o',...
    'MarkerSize',8,'MarkerFaceColor','m');
plot3(X(502:668,1),X(502:668,2),CXTrue(502:668),'o',...
    'MarkerSize',8,'MarkerFaceColor','g');
plot3(X(669:835,1),X(669:835,2),CXTrue(669:835),'o',...
    'MarkerSize',8,'MarkerFaceColor','y');
legend({'2925','3025','3125','3225','3325'},...
    'Location','eastoutside',...
    'Orientation','vertical',...
    'FontSize',13,'FontWeight','bold')
mesh(x,y,CX);
xlabel('S/X','FontSize',13,'FontWeight','bold')
ylabel('T-t','FontSize',13,'FontWeight','bold')
zlabel('C/X','FontSize',13,'FontWeight','bold')
% axis([0.8 1.5 0 0.6 0 0.4]);
grid on
grid minor
hold off
%% draw validation
CXpred=ones(LAll,1);
for i=1:LAll
    CXpred(i)=w(1)*sqrt((XAll(i,:)-m1)*C1*(XAll(i,:)-m1)')...
        +w(2)*sqrt((XAll(i,:)-m2)*C2*(XAll(i,:)-m2)')...
        +w(3)*sqrt((XAll(i,:)-m3)*C3*(XAll(i,:)-m3)')...
        +w(4)*sqrt((XAll(i,:)-m4)*C4*(XAll(i,:)-m4)')...
        +XAll(i,:)*[w(5);w(6)]+w(7);
end

figure(3),clf,
xx1=1:835;
plot(xx1,CXpred,'r','LineWidth',1.5);
xlabel('Date','FontSize',13,'FontWeight','bold')
ylabel('C/X','FontSize',13,'FontWeight','bold')
hold on
plot(xx1,CXtrueAll,'b','LineWidth',1.5);
axis([-inf,inf,-inf,inf]);
legend({'predicted','real'},'Location','northwest',...
    'FontSize',13,'FontWeight','bold');
% plot([175,175],[0,0.12],'k','LineWidth',2);
grid on
grid minor
hold off
%% draw scatter
figure(4),clf,
plot3(X(1:167,1),X(1:167,2),CXTrue(1:167),'o',...
    'MarkerSize',8,'MarkerFaceColor','b');
hold on
plot3(X(168:334,1),X(168:334,2),CXTrue(168:334),'o',...
    'MarkerSize',8,'MarkerFaceColor','r');
plot3(X(335:501,1),X(335:501,2),CXTrue(335:501),'o',...
    'MarkerSize',8,'MarkerFaceColor','m');
plot3(X(502:668,1),X(502:668,2),CXTrue(502:668),'o',...
    'MarkerSize',8,'MarkerFaceColor','g');
plot3(X(669:835,1),X(669:835,2),CXTrue(669:835),'o',...
    'MarkerSize',8,'MarkerFaceColor','y');
legend({'2925','3025','3125','3225','3325'},...
    'Location','eastoutside',...
    'Orientation','vertical',...
    'FontSize',13,'FontWeight','bold')

xlabel('S/X','FontSize',13,'FontWeight','bold')
ylabel('T-t','FontSize',13,'FontWeight','bold')
zlabel('C/X','FontSize',13,'FontWeight','bold')
grid on
grid minor
hold off
%%



