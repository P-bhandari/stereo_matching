yNoc0 = [-nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan] ;
yOcc0 = [-nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan] ;
sgbmNoc0 = [-nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan] ;
sgbmOcc0 = [-nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan; -nan -nan] ;
-nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan] ;
yOcc0 = [-nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan] ;
sgbmNoc0 = [-nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan] ;
sgbmOcc0 = [-nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan; -nan -nan -nan -nan -nan] ;

x = [1:1:13] ; 
 figure() ; 
 subplot(2,1,1); 
 bar(x,yOcc0);
 grid on ; 
title('CannyEdgeOcc0');

ax=gca; 
 ax.XTickLabel = {'1Edge', '2Edge','4Edge','8Edge','12Edge', '16Edge','16PointAl','16PointDia','12Point', '8PointAl','4PointDia','2point','1point'} 
ax.XTickLabelRotation=45;
subplot(2,1,2); 
 bar(x,sgbmOcc0);
 grid on ; 
title('SGBMOcc0');

ax=gca; 
 ax.XTickLabel = {'1Edge', '2Edge','4Edge','8Edge','12Edge', '16Edge','16PointAl','16PointDia','12Point', '8PointAl','4PointDia','2point','1point'} 
ax.XTickLabelRotation=45;
