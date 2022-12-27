load IrisDataAnnotated X I


% Locating indces and rearranging vectors 
%As per species 
I_1 = find (I == 1); % setosa 
I_2 = find(I == 2); % veriscolor 
I_3 = find(I == 3); % virginca 
II = [I_1, I_2, I_3];
Y = X(:,II);
y_c = 1/length(II)*sum(Y,2);%centering data
Y_c = Y - y_c*ones(1,length(II));

%PCA
[U,D,V] = svd(Y_c);

% First three principle components in scatter plot 

figure(1)
Z_1 = U(:,1:3)'*Y_c(:,1:length(I_1)); %principle components for iris
plot3(Z_1(1,:),Z_1(2,:),Z_1(3,:),'r.','MarkerSize',5); % plotting iris

hold on 
Z_2 = U(:,1:3)'*Y_c(:,length(I_1)+1:length(I_1)+length(I_2));
plot3(Z_2(1,:),Z_2(2,:),Z_2(3,:),'b.','MarkerSize',5);

Z_3 = U(:,1:3)'*Y_c(:,length(I_1)+length(I_2)+1:length(II));
plot3(Z_3(1,:),Z_3(2,:),Z_3(3,:),'g.','MarkerSize',5);


set(gca,'FontSize',15)
xlabel('1st rows');
ylabel('2nd rows')

