load WineData-1.mat X I

% arranging the data into submatrices
I_assign = I';
I_1 = find (I_assign == 1);  
I_2 = find(I_assign == 2);  
I_3 = find(I_assign == 3); 
II = [I_1, I_2, I_3];

Y = X(:,II);

% PCA for winedata 

c = (1/size(Y,2))*sum(Y,2); 

Y_c = Y - c.*ones(1, length(II));

%PCA
[U,D,V] = svd(Y_c);

% First three principle components in scatter plot 

figure(1)
Z_1 = U(:,1:3)'*Y_c(:,1:length(I_1)); 
plot3(Z_1(1,:),Z_1(2,:),Z_1(3,:),'r.','MarkerSize',5); 

hold on 
Z_2 = U(:,1:3)'*Y_c(:,length(I_1)+1:length(I_1)+length(I_2));
plot3(Z_2(1,:),Z_2(2,:),Z_2(3,:),'r.','MarkerSize',5);

Z_3 = U(:,1:3)'*Y_c(:,length(I_1)+length(I_2)+1:length(II'));
plot3(Z_3(1,:),Z_3(2,:),Z_3(3,:),'r.','MarkerSize',5);


set(gca,'FontSize',15)
xlabel('1st rows');
ylabel('2nd rows')

% computing the cluster means for each submatrix 

CY_1 = (1/length(I_1))*sum(Y(:,1:length(I_1)),2);
CY_2 = (1/length(I_2))*sum(Y(:,length(I_1)+1:length(I_1)+length(I_2)),2);
CY_3 = (1/length(I_3))*sum(Y(:,length(I_1)+length(I_2)+1:length(II')),2);

% Within cluster centering and scattering matrix

SX_1 = Y(:,1:length(I_1)) - CY_1;
SX_2 = Y(:,length(I_1)+1:length(I_1)+length(I_2)) - CY_2;
SX_3 = Y(:,length(I_1)+length(I_2)+1:length(II')) - CY_3;

Xw = [SX_1, SX_2, SX_3];

Sw = Xw*Xw';

% Between cluster centering and scattering matrix 

Xj = zeros(13,178);
Xj(:,1:length(I_1)) = CY_1.*ones(13,length(I_1));
Xj(:,length(I_1)+1:length(I_1)+length(I_2)) = CY_2.*ones(13,length(I_2));
Xj(:,length(I_1)+length(I_2)+1:length(II')) = CY_3.*ones(13,length(I_3));

Xb = Xj - c.*ones(13,178);

Sb = Xb*Xb';

% Largest eigenvalue/vectors and Cholesky factorization

k = chol(Sw);
kswk = inv(k')*Sb*inv(k);
[V,d1] = eig(kswk);

% computing the 1st two LDA directions 

q1 = inv(k)*V(:,1);
q2 = inv(k)*V(:,2);

% Computing the LDA components of original data and plotting 

Z1 = q1'*X; 
Z2 = q2'*X; 

figure(2)
plot(Z1, Z2, 'k.','MarkerSize',5);

[v, k] = max(q1);
k
[v, k] = max(q2);
k

