function [] = plot_mesh(X, Y, Z)   
    F = TriScatteredInterp(X, Y, Z);
    rx = (max(X) - min(X)) / 30;
    ry = (max(Y) - min(Y)) / 30;
    [qx,qy] = meshgrid(min(X):rx:max(X), min(Y):ry:max(Y));
    qz = F(qx,qy);
    
    
    plot3(X(1:167),Y(1:167),Z(1:167),'o',...
        'MarkerSize',6,'MarkerFaceColor','b');
    hold on;
    plot3(X(168:334),Y(168:334),Z(168:334),'o',...
        'MarkerSize',6,'MarkerFaceColor','r');
    plot3(X(335:501),Y(335:501),Z(335:501),'o',...
        'MarkerSize',6,'MarkerFaceColor','m');
    plot3(X(502:668),Y(502:668),Z(502:668),'o',...
        'MarkerSize',6,'MarkerFaceColor','g');
    plot3(X(669:835),Y(669:835),Z(669:835),'o',...
        'MarkerSize',6,'MarkerFaceColor','y');
    legend({'2925','3025','3125','3225','3325'},...
        'Location','eastoutside',...
        'Orientation','vertical',...
        'FontSize',13,'FontWeight','bold')
    
    mesh(qx,qy,qz);

    set(gca,'FontSize',13)
    
    grid on
    grid minor
    hold off
end

