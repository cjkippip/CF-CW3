% Computational Finance CW2
% Question 1
% train singal file
load options.mat
%%
MSE=ones(5,1);
MSE2=ones(5,1);
for optionNum=1:5
% optionNum=5;% number of strike price
strikePrices=[2925 3025 3125 3225 3325];

L=length(stockPrice);% data length 222
LWin=fix(L/4);% window length 55
LUse=L-LWin;% used length 167

LAll=167; % all data length
LTrain=round(0.8*LUse); % train data length
LTest=LAll-LTrain; % test data length 

interval_All=(optionNum-1)*LUse+1:(optionNum-1)*LUse+LUse;
interval_Tr=(optionNum-1)*LUse+1:(optionNum-1)*LUse+LTrain;
interval_Ts=(optionNum-1)*LUse+LTrain+1:(optionNum-1)*LUse+LUse;

Tt=ones(L,1);
for i=1:L
    Tt(i)=(L-i)/222;
end

% data: X=[S/X T-t]
X=[stockPrice(56:L)/strikePrices(1) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(2) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(3) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(4) Tt(56:L,1);...
    stockPrice(56:L)/strikePrices(5) Tt(56:L,1)];

XAll=X(interval_All,:); % all data
% XTrain=X(interval_Tr,:); % train data
% XTest=X(interval_Ts,:); % test data
XTrain=XAll(1:LTrain,:); % train data
XTest=XAll(LTrain+1:end,:); % test data

% normalized call option price from BS formula
CX_BS=[BSOptionCPrices(:,1)./strikePrices(1);...
    BSOptionCPrices(:,2)./strikePrices(2);...
    BSOptionCPrices(:,3)./strikePrices(3);...
    BSOptionCPrices(:,4)./strikePrices(4);...
    BSOptionCPrices(:,5)./strikePrices(5)];

CX_BS_All=CX_BS(interval_All,:); % all tag
% CX_BS_Train=CX_BS(interval_Tr,:); % train tag
% CX_BS_Test=CX_BS(interval_Ts,:); % test tag
CX_BS_Train=CX_BS_All(1:LTrain,:); % train tag
CX_BS_Test=CX_BS_All(LTrain+1:end,:); % test tag

%% GMModel generate 4 means and covariances
GMModel = fitgmdist(XTrain,4,'RegularizationValue',0);
% figure(1),clf,
% scatter(XTrain(:,1),XTrain(:,2),'r.');
% hold on
% ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
% title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
% axis([0.8 1.3 -0.1 0.8]);
% xlabel('S/X','FontSize',13,'FontWeight','bold');
% ylabel('T-t','FontSize',13,'FontWeight','bold');
% grid on
% grid minor
% hold off
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
minimize( norm(designMat*w-CX_BS_Train) )
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

% figure(2),clf,
% plot3(X(interval_All,1),X(interval_All,2),CX_BS(interval_All),...
%     'o','MarkerSize',8,'MarkerFaceColor','b');
% hold on
% mesh(x,y,CX);
% xlabel('S/X','FontSize',13,'FontWeight','bold')
% ylabel('T-t','FontSize',13,'FontWeight','bold')
% zlabel('C/X','FontSize',13,'FontWeight','bold')
% axis([0.82 1.15 0 0.7 -0.01 0.17]);
% grid on
% grid minor
% hold off
%% draw validation
CXpred=ones(LUse,1);
for i=1:LUse
    CXpred(i)=w(1)*sqrt((XAll(i,:)-m1)*C1*(XAll(i,:)-m1)')...
        +w(2)*sqrt((XAll(i,:)-m2)*C2*(XAll(i,:)-m2)')...
        +w(3)*sqrt((XAll(i,:)-m3)*C3*(XAll(i,:)-m3)')...
        +w(4)*sqrt((XAll(i,:)-m4)*C4*(XAll(i,:)-m4)')...
        +XAll(i,:)*[w(5);w(6)]+w(7);
end
preMax=max(CXpred);
preMin=min(CXpred);
BSMax=max(CX_BS_All);
BSMin=min(CX_BS_All);
tMax=max(preMax,BSMax);
tMin=min(preMin,BSMin);

MSE(optionNum)=norm(CXpred(LTrain+1:end)-...
    CX_BS_All(LTrain+1:end))/LTest;
MSE2(optionNum)=norm(CXpred(LTrain+1:end)*strikePrices(optionNum)-...
    CX_BS_All(LTrain+1:end)*strikePrices(optionNum))/LTest;

figure(optionNum),clf,
xx1=56:222;
plot(xx1,CXpred,'r','LineWidth',1.5);
title(['File ',num2str(optionNum)],'FontSize',16)
xlabel('Date','FontSize',13,'FontWeight','bold')
ylabel('C/X','FontSize',13,'FontWeight','bold')
hold on
plot(xx1,CX_BS_All,'b','LineWidth',1.5);
legend({'predicted','real'},'Location','northwest',...
    'FontSize',13,'FontWeight','bold');
plot([LTrain+LWin,LTrain+LWin],[-0.015,0.15],'k','LineWidth',2);
axis([-inf inf -0.015 0.15]);
set(gca,'FontSize',13)
grid on
grid minor
hold off
end
%%

%% draw scatter
% figure(4),clf,
% plot3(X(1:167,1),X(1:167,2),CX_BS(1:167),'o',...
%     'MarkerSize',6,'MarkerFaceColor','b');
% hold on
% plot3(X(168:334,1),X(168:334,2),CX_BS(168:334),'o',...
%     'MarkerSize',6,'MarkerFaceColor','r');
% plot3(X(335:501,1),X(335:501,2),CX_BS(335:501),'o',...
%     'MarkerSize',6,'MarkerFaceColor','m');
% plot3(X(502:668,1),X(502:668,2),CX_BS(502:668),'o',...
%     'MarkerSize',6,'MarkerFaceColor','g');
% plot3(X(669:835,1),X(669:835,2),CX_BS(669:835),'o',...
%     'MarkerSize',6,'MarkerFaceColor','y');
% legend({'2925','3025','3125','3225','3325'},...
%     'Location','eastoutside',...
%     'Orientation','vertical',...
%     'FontSize',13,'FontWeight','bold')
% title('data generated by BS formula','FontSize',16)
% xlabel('S/X','FontSize',13,'FontWeight','bold')
% ylabel('T-t','FontSize',13,'FontWeight','bold')
% zlabel('C/X','FontSize',13,'FontWeight','bold')
% grid on
% grid minor
% hold off
%% delta
% [CallDelta,~]=blsdelta(S,K,r,T,Vol,Yield);













