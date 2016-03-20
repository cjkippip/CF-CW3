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
mu1 = [1 2];
Sigma1 = [2 0; 0 0.5];
mu2 = [-3 -5];
Sigma2 = [1 0;0 1];
rng(1); % For reproducibility
X = [mvnrnd(mu1,Sigma1,1000);mvnrnd(mu2,Sigma2,1000)];

GMModel = fitgmdist(X,2);

figure
y = [zeros(1000,1);ones(1000,1)];
h = gscatter(X(:,1),X(:,2),y);
hold on
ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
legend(h,'Model 0','Model1')
hold off
%%
load discrim
figure;
gscatter(ratings(:,1),ratings(:,2),group,'br','xo')
xlabel('climate');
ylabel('housing');
%%
xx=1:10;
yy=xx.^2;
plot(xx,yy);










