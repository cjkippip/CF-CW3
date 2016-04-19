% figure(1),clf,
% plot(x1,y1,'b','LineWidth',2);
% title('Profit of asset price','FontSize',15)
% ylabel('Profit','FontSize',13,'FontWeight','bold')
% xlabel('Asset price','FontSize',13,'FontWeight','bold')
% hold on
% plot(x2,y2,'b','LineWidth',2);
% plot(x3,y3,'b','LineWidth',2);
% plot(xx1,yy1,'b.','LineWidth',2);
% plot(xx2,yy2,'b.','LineWidth',2);
% axis([0 85 0 45])
% grid on
% hold off
%%
% mu1 = [1 2];
% Sigma1 = [2 0; 0 0.5];
% mu2 = [-3 -5];
% Sigma2 = [1 0;0 1];
% rng(1); % For reproducibility
% X = [mvnrnd(mu1,Sigma1,1000);mvnrnd(mu2,Sigma2,1000)];
% 
% GMModel = fitgmdist(X,2);
% 
% figure
% y = [zeros(1000,1);ones(1000,1)];
% h = gscatter(X(:,1),X(:,2),y);
% hold on
% ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
% title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
% legend(h,'Model 0','Model1')
% hold off
%%
% load discrim
% figure;
% gscatter(ratings(:,1),ratings(:,2),group,'br','xo')
% xlabel('climate');
% ylabel('housing');
%%
% xx=1:10;
% yy=xx.^2;
% plot(xx,yy);
%% aboutezplot 
% syms x1 x2 CX
%%
CXNum=30;
x=linspace(0.7,1.5,CXNum);
% x=linspace(1.5,0.7,CXNum);
y=linspace(0.7,0,CXNum);
% y=linspace(0,0.7,CXNum);

m1=[1.35 0.45];
m2=[1.18 0.24];
m3=[0.98 -0.2];
m4=[1.05 -0.1];
C1=[59.79 -0.03;-0.03 10.24];
C2=[59.79 -0.03;-0.03 10.24];
C3=[59.79 -0.03;-0.03 10.24];
C4=[59.79 -0.03;-0.03 10.24]; 

CX=ones(CXNum,CXNum);

for i=1:CXNum
    for j=1:CXNum
        CX(j,i)=-0.06*sqrt(([x(i),y(j)]-m1)*C1*([x(i),y(j)]-m1)'+2.55)...
            -0.03*sqrt(([x(i),y(j)]-m2)*C2*([x(i),y(j)]-m2)'+1.97)...
            +0.03*sqrt(([x(i),y(j)]-m3)*C3*([x(i),y(j)]-m3)')...
            +0.10*sqrt(([x(i),y(j)]-m4)*C4*([x(i),y(j)]-m4)'+1.62)...
            +[x(i),y(j)]*[0.14;-0.24]-0.01;
    end
end

figure(1),clf,
mesh(x,y,CX);
xlabel('S/X','FontSize',13,'FontWeight','bold')
ylabel('T-t','FontSize',13,'FontWeight','bold')
zlabel('C/X','FontSize',13,'FontWeight','bold')
% axis([0.8 1.5 0 0.6 0 0.4]);
grid on
grid minor
%%
aa=designMat*w;
aaa=norm(aa-CXtrueTrain);
%%
S=49;
K=50;
r=0.05;
T=0.3846;
Vol=0.13;
y=blsprice(S,K,r,T,Vol);

















