% Computational Finance CW2
% Question 1
% train 5 files
load options.mat

% optionNum=4;% number of strike price
strikePrices=[2925 3025 3125 3225 3325];

L=length(stockPrice);% data length 222
LWin=fix(L/4);% window length 55
LUse=L-LWin;% used length 167
%%
L_All=LUse;
L_Tr=round(0.98*L_All); 
L_Ts=L_All-L_Tr;

LAll=5*L_All; % all data length
LTrain=5*L_Tr; % train data length
LTest=5*L_Ts; % test data length 

% interval_Tr=(optionNum-1)*Lrest+1:(optionNum-1)*Lrest+LTrain;
% interval_Ts=(optionNum-1)*Lrest+LTrain+1:(optionNum-1)*Lrest+Lrest;
% interval_All=(optionNum-1)*Lrest+1:(optionNum-1)*Lrest+Lrest;

interval_All=1:835;
interval_Tr=[1:L_Tr,...
    L_All+1:L_All+L_Tr,...
    2*L_All+1:2*L_All+L_Tr,...
    3*L_All+1:3*L_All+L_Tr,...
    4*L_All+1:4*L_All+L_Tr];
interval_Ts=[L_Tr+1:L_All,...
    L_All+L_Tr+1:2*L_All,...
    2*L_All+L_Tr+1:3*L_All,...
    3*L_All+L_Tr+1:4*L_All,...
    4*L_All+L_Tr+1:5*L_All];

Tt=ones(L,1);
for i=1:L
    Tt(i)=(L-i)/252;
end

% data: X=[S/X T-t]
X=[stockPrice(56:L)./strikePrices(1) Tt(56:L,1);...
    stockPrice(56:L)./strikePrices(2) Tt(56:L,1);...
    stockPrice(56:L)./strikePrices(3) Tt(56:L,1);...
    stockPrice(56:L)./strikePrices(4) Tt(56:L,1);...
    stockPrice(56:L)./strikePrices(5) Tt(56:L,1)];
XAll=X(interval_All,:); % all data
XTrain=X(interval_Tr,:); % train data
XTest=X(interval_Ts,:); % test data

% normalized call option price from BS formula
CX_BS=[BSOptionCPrices(:,1)./strikePrices(1);...
    BSOptionCPrices(:,2)./strikePrices(2);...
    BSOptionCPrices(:,3)./strikePrices(3);...
    BSOptionCPrices(:,4)./strikePrices(4);...
    BSOptionCPrices(:,5)./strikePrices(5)];
CX_BS_All=CX_BS(interval_All,:); % all tag
CX_BS_Train=CX_BS(interval_Tr,:); % train tag
CX_BS_Test=CX_BS(interval_Ts,:); % test tag

% normalized call option price from observation
CX_ob=[optionCPrice(56:222,1)./strikePrices(1);...
    optionCPrice(56:222,2)./strikePrices(2);...
    optionCPrice(56:222,3)./strikePrices(3);...
    optionCPrice(56:222,4)./strikePrices(4);...
    optionCPrice(56:222,5)./strikePrices(5)];
CX_ob_All=CX_ob(interval_All,:); % all tag
CX_ob_Train=CX_ob(interval_Tr,:); % train tag
CX_ob_Test=CX_ob(interval_Ts,:); % test tag

% data with tag
Y_BS=[X CX_BS];
Y_ob=[X CX_ob];


%% GMModel generate 4 means and covariances
GMModel = fitgmdist(XTrain,4,'RegularizationValue',0);

figure(1),clf,
scatter(XTrain(:,1),XTrain(:,2),'r.');
hold on
ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
axis([0.8 1.3 -0.1 0.8]);
xlabel('S/X','FontSize',13,'FontWeight','bold');
ylabel('T-t','FontSize',13,'FontWeight','bold');
grid on
grid minor
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
minimize( norm(designMat*w-CX_BS_Train) )
cvx_end
%% C/X surface
% CXNum=60;
% x=linspace(0.82,1.15,CXNum);
% y=linspace(0.7,0,CXNum);
step1=0.01;
step2=0.02;
xx1=0.82:step1:1.15;
yy1=0:step2:0.7;
length_SX=length(xx1);
length_Tt=length(yy1);

CX=ones(length_Tt,length_SX);
for i=1:length_SX
    for j=1:length_Tt
        CX(j,i)=w(1)*sqrt(([xx1(i),yy1(j)]-m1)*C1*([xx1(i),yy1(j)]-m1)')...
            +w(2)*sqrt(([xx1(i),yy1(j)]-m2)*C2*([xx1(i),yy1(j)]-m2)')...
            +w(3)*sqrt(([xx1(i),yy1(j)]-m3)*C3*([xx1(i),yy1(j)]-m3)')...
            +w(4)*sqrt(([xx1(i),yy1(j)]-m4)*C4*([xx1(i),yy1(j)]-m4)')...
            +[xx1(i),yy1(j)]*[w(5);w(6)]+w(7);
    end
end
%% draw C/X surface and BS scatter
figure(2),clf,
plot3(X(1:167,1),X(1:167,2),CX_BS(1:167),'o',...
    'MarkerSize',6,'MarkerFaceColor','b');
hold on
plot3(X(168:334,1),X(168:334,2),CX_BS(168:334),'o',...
    'MarkerSize',6,'MarkerFaceColor','r');
plot3(X(335:501,1),X(335:501,2),CX_BS(335:501),'o',...
    'MarkerSize',6,'MarkerFaceColor','m');
plot3(X(502:668,1),X(502:668,2),CX_BS(502:668),'o',...
    'MarkerSize',6,'MarkerFaceColor','g');
plot3(X(669:835,1),X(669:835,2),CX_BS(669:835),'o',...
    'MarkerSize',6,'MarkerFaceColor','y');
legend({'2925','3025','3125','3225','3325'},...
    'Location','eastoutside',...
    'Orientation','vertical',...
    'FontSize',13,'FontWeight','bold')
mesh(xx1,yy1,CX);

title('C/X surface and BS scatter','FontSize',16)
xlabel('S/X','FontSize',13,'FontWeight','bold')
ylabel('T-t','FontSize',13,'FontWeight','bold')
zlabel('C/X','FontSize',13,'FontWeight','bold')
axis([0.82 1.15 0 0.7 -0.01 0.17]);
grid on
grid minor
hold off
%% draw C/X surface and observed scatter
% figure(3),clf,
% plot3(X(1:167,1),X(1:167,2),CX_ob(1:167),'o',...
%     'MarkerSize',8,'MarkerFaceColor','b');
% hold on
% plot3(X(168:334,1),X(168:334,2),CX_ob(168:334),'o',...
%     'MarkerSize',8,'MarkerFaceColor','r');
% plot3(X(335:501,1),X(335:501,2),CX_ob(335:501),'o',...
%     'MarkerSize',8,'MarkerFaceColor','m');
% plot3(X(502:668,1),X(502:668,2),CX_ob(502:668),'o',...
%     'MarkerSize',8,'MarkerFaceColor','g');
% plot3(X(669:835,1),X(669:835,2),CX_ob(669:835),'o',...
%     'MarkerSize',8,'MarkerFaceColor','y');
% legend({'2925','3025','3125','3225','3325'},...
%     'Location','eastoutside',...
%     'Orientation','vertical',...
%     'FontSize',13,'FontWeight','bold')
% mesh(xx1,yy1,CX);
% 
% title('C/X surface and observed scatter','FontSize',16)
% xlabel('S/X','FontSize',13,'FontWeight','bold')
% ylabel('T-t','FontSize',13,'FontWeight','bold')
% zlabel('C/X','FontSize',13,'FontWeight','bold')
% axis([0.82 1.15 0 0.7 -0.05 0.17]);
% grid on
% grid minor
% hold off
%% draw validation
CXpred=ones(LAll,1);
for i=1:LAll
    CXpred(i)=w(1)*sqrt((XAll(i,:)-m1)*C1*(XAll(i,:)-m1)')...
        +w(2)*sqrt((XAll(i,:)-m2)*C2*(XAll(i,:)-m2)')...
        +w(3)*sqrt((XAll(i,:)-m3)*C3*(XAll(i,:)-m3)')...
        +w(4)*sqrt((XAll(i,:)-m4)*C4*(XAll(i,:)-m4)')...
        +XAll(i,:)*[w(5);w(6)]+w(7);
end
%% draw validation
figure(4),clf,
x1=1:835;
plot(x1,CXpred,'r','LineWidth',1.5);
xlabel('Date','FontSize',13,'FontWeight','bold')
ylabel('C/X','FontSize',13,'FontWeight','bold')
hold on
plot(x1,CX_BS_All,'b','LineWidth',1.5);
axis([-inf,inf,-inf,inf]);
legend({'predicted','real'},'Location','northwest',...
    'FontSize',13,'FontWeight','bold');
plot([LTrain,LTrain],[-0.05,0.15],'k','LineWidth',2);
grid on
grid minor
hold off
%% draw scatter BS
figure(5),clf,
plot3(X(1:167,1),X(1:167,2),CX_BS(1:167),'o',...
    'MarkerSize',6,'MarkerFaceColor','b');
hold on
plot3(X(168:334,1),X(168:334,2),CX_BS(168:334),'o',...
    'MarkerSize',6,'MarkerFaceColor','r');
plot3(X(335:501,1),X(335:501,2),CX_BS(335:501),'o',...
    'MarkerSize',6,'MarkerFaceColor','m');
plot3(X(502:668,1),X(502:668,2),CX_BS(502:668),'o',...
    'MarkerSize',6,'MarkerFaceColor','g');
plot3(X(669:835,1),X(669:835,2),CX_BS(669:835),'o',...
    'MarkerSize',6,'MarkerFaceColor','y');
legend({'2925','3025','3125','3225','3325'},...
    'Location','eastoutside',...
    'Orientation','vertical',...
    'FontSize',13,'FontWeight','bold')

title('Data generated by BS formula','FontSize',16)
xlabel('S/X','FontSize',13,'FontWeight','bold')
ylabel('T-t','FontSize',13,'FontWeight','bold')
zlabel('C/X','FontSize',13,'FontWeight','bold')
grid on
grid minor
hold off
%% draw scatter observation
% figure(6),clf,
% plot3(X(1:167,1),X(1:167,2),CX_ob(1:167),'o',...
%     'MarkerSize',8,'MarkerFaceColor','b');
% hold on
% plot3(X(168:334,1),X(168:334,2),CX_ob(168:334),'o',...
%     'MarkerSize',8,'MarkerFaceColor','r');
% plot3(X(335:501,1),X(335:501,2),CX_ob(335:501),'o',...
%     'MarkerSize',8,'MarkerFaceColor','m');
% plot3(X(502:668,1),X(502:668,2),CX_ob(502:668),'o',...
%     'MarkerSize',8,'MarkerFaceColor','g');
% plot3(X(669:835,1),X(669:835,2),CX_ob(669:835),'o',...
%     'MarkerSize',8,'MarkerFaceColor','y');
% legend({'2925','3025','3125','3225','3325'},...
%     'Location','eastoutside',...
%     'Orientation','vertical',...
%     'FontSize',13,'FontWeight','bold')
% 
% title('Observed data','FontSize',16)
% xlabel('S/X','FontSize',13,'FontWeight','bold')
% ylabel('T-t','FontSize',13,'FontWeight','bold')
% zlabel('C/X','FontSize',13,'FontWeight','bold')
% grid on
% grid minor
% hold off
%% delta
delta=diff(CX,1,2)/step1;
figure(7),clf,
mesh(xx1(1:end-1),yy1,delta);
title('Delta surface and BS scatter','FontSize',16)
xlabel('S/X','FontSize',13,'FontWeight','bold')
ylabel('T-t','FontSize',13,'FontWeight','bold')
zlabel('delta','FontSize',13,'FontWeight','bold')
grid on
grid minor







