% Computational Finance CW2
% Question 1
% train 5 files
% every file train and test
load options.mat
%%
strikePrices=[2925 3025 3125 3225 3325];

L=length(stockPrice);% data length 222
LWin=fix(L/4);% window length 55
LUse=L-LWin;% used length 167

rate_Tr=0.8;

L_All=LUse;
L_Tr=round(rate_Tr*L_All); 
L_Ts=L_All-L_Tr;

LAll=5*L_All; % all data length 835
LTrain=5*L_Tr; % train data length 670
LTest=5*L_Ts; % test data length 165

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

Tt=ones(L,1); % 222
for i=1:L
    Tt(i)=(L-i)/252;
end
Tt2=Tt(56:L,1); % 167

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

% Delta
Delta_BS_C=ones(L_All,5);
for j=1:5
    K=strikePrices(j);
    for i=1:LUse  
        S=stockPrice(LWin+i);      
        r=0.06;
        T=(LUse-i+1)/252;
        [Delta_BS_C(i,j),~]=blsdelta(S,K,r,T,hisVols(i));      
    end
end
Delta_BS=[Delta_BS_C(:,1);Delta_BS_C(:,2);...
    Delta_BS_C(:,3);Delta_BS_C(:,4);Delta_BS_C(:,5)];
Delta_BS_Tr=Delta_BS(interval_Tr,:);
Delta_BS_Ts=Delta_BS(interval_Ts,:);

%% GMModel generate 4 means and covariances
GMModel = fitgmdist(XTrain,4,'RegularizationValue',0);

% figure(1),clf,
% scatter(XTrain(:,1),XTrain(:,2),'r.');
% hold on
% ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
% title('{\bf Scatter Plot and Fitted Mixture Gaussian Contours}',...
%     'FontSize',16)
% axis([0.80 1.2 -0.1 0.8]);
% xlabel('S/X','FontSize',16,'FontWeight','bold');
% ylabel('T-t','FontSize',16,'FontWeight','bold');
% set(gca,'FontSize',13)
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
%% scattered interpolant surface
figure(2),clf,
plot_mesh(X(:,1),X(:,2),CX_BS);
title('Simulated(BS) C/X data points and scattered interpolant surface','FontSize',16)
xlabel('S/X','FontSize',16,'FontWeight','bold')
ylabel('T-t','FontSize',16,'FontWeight','bold')
zlabel('C/X','FontSize',16,'FontWeight','bold')

figure(3),clf,
plot_mesh(X(:,1),X(:,2),Delta_BS);
title('Simulated(BS) Delta data points and scattered interpolant surface','FontSize',16)
xlabel('S/X','FontSize',16,'FontWeight','bold')
ylabel('T-t','FontSize',16,'FontWeight','bold')
zlabel('Delta','FontSize',16,'FontWeight','bold')
%%
step1=0.01;
step2=-0.005;
xx1=0.82:step1:1.15; % S/X direction
yy1=0.8:step2:0; % T-t direction
length_SX=length(xx1);
length_Tt=length(yy1);
yy1_Tr=yy1(1:round(rate_Tr*length_Tt));
yy1_Ts=yy1(round(rate_Tr*length_Tt)+1:end);
length_Tt_Tr=length(yy1_Tr);
length_Tt_Ts=length(yy1_Ts);

CX_RBF=ones(length_Tt,length_SX);
for i=1:length_SX
    for j=1:length_Tt
        CX_RBF(j,i)=w(1)*sqrt(([xx1(i),yy1(j)]-m1)*C1*([xx1(i),yy1(j)]-m1)')...
            +w(2)*sqrt(([xx1(i),yy1(j)]-m2)*C2*([xx1(i),yy1(j)]-m2)')...
            +w(3)*sqrt(([xx1(i),yy1(j)]-m3)*C3*([xx1(i),yy1(j)]-m3)')...
            +w(4)*sqrt(([xx1(i),yy1(j)]-m4)*C4*([xx1(i),yy1(j)]-m4)')...
            +[xx1(i),yy1(j)]*[w(5);w(6)]+w(7);
    end
end
Delta_RBF=diff(CX_RBF,1,2)/step1;
%% draw RBF C/X surface
figure(4),clf,
mesh(xx1,yy1_Ts,CX_RBF(length_Tt_Tr+1:end,:));
title('RBF C/X surface',...
    'FontSize',16)
xlabel('S/X','FontSize',16,'FontWeight','bold')
ylabel('T-t','FontSize',16,'FontWeight','bold')
zlabel('C/X','FontSize',16,'FontWeight','bold')
% axis([0.82 1.15 -inf inf -0.01 0.17]);
set(gca,'FontSize',13)
grid on
grid minor
hold off
%% draw RBF Delta surface
figure(5),clf,
mesh(xx1(2:end),yy1_Ts,Delta_RBF(length_Tt_Tr+1:end,:));
title('RBF Delta surface',...
    'FontSize',16)
xlabel('S/X','FontSize',16,'FontWeight','bold')
ylabel('T-t','FontSize',16,'FontWeight','bold')
zlabel('Delta','FontSize',16,'FontWeight','bold')
% axis([0.82 1.15 -inf inf -0.01 0.17]);
set(gca,'FontSize',13)
grid on
grid minor
hold off
%% C/X and Delta error surface 
F_CX = scatteredInterpolant(X(:,1),X(:,2),CX_BS);
F_Delta = scatteredInterpolant(X(:,1),X(:,2),Delta_BS);

[xx2,yy2]=meshgrid(xx1,yy1_Ts);
[xx3,yy3]=meshgrid(xx1(2:end),yy1_Ts);

CX_BS_grid=F_CX(xx2,yy2);
Delta_BS_grid=F_Delta(xx3,yy3);

err_CX_grid=CX_RBF(length_Tt_Tr+1:end,:)-CX_BS_grid;
err_Delta_grid=Delta_RBF(length_Tt_Tr+1:end,:)-Delta_BS_grid;
%% C/X error surface 
figure(6),clf,
mesh(xx2,yy2,err_CX_grid);
title('C/X error',...
    'FontSize',16)
xlabel('S/X','FontSize',16,'FontWeight','bold')
ylabel('T-t','FontSize',16,'FontWeight','bold')
zlabel('C/X','FontSize',16,'FontWeight','bold')
% axis([0.82 1.15 -inf inf -0.01 0.17]);
set(gca,'FontSize',13)
grid on
grid minor
hold off
%% Delta error surface 
figure(7),clf,
mesh(xx3,yy3,err_Delta_grid);
title('Delta error',...
    'FontSize',16)
xlabel('S/X','FontSize',16,'FontWeight','bold')
ylabel('T-t','FontSize',16,'FontWeight','bold')
zlabel('Delta','FontSize',16,'FontWeight','bold')
% axis([0.82 1.15 -inf inf -0.01 0.17]);
set(gca,'FontSize',13)
grid on
grid minor
hold off














