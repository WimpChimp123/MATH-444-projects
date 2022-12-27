%% Running k_medoids 

load WineData.mat X 

% Picking 3 random characteristic vectors
    
C_rand = [88, 143, 55];
 
% Distance matrix 

D = zeros(size(X,2),size(X,2)); 
dd = zeros(1,size(X,2));
for i = 1:size(X,2)
    
    dd = zeros(1,size(X,2));
    
    for j = 1:size(X,2)
        dd(1,j) = norm(X(:,j) - X(:,i));
  
    end 

    D(i,:) = dd(1,:);
  
end 

% Running k_medoids 
Tau = 0.01; 

[I_assign, I_bar] = my_k_medoids(C_rand,D,Tau);

% Plotting the data 

I1 = find (I_assign == 1); % setosa 
I2 = find(I_assign == 2); % veriscolor 
I3 = find(I_assign == 3); % virginca 
II = [I1, I2, I3];
Y = X(:,II);
y_c = 1/length(II)*sum(Y,2);%centering data
Y_c = Y - y_c*ones(1,length(II));

%Selecting clusters Using PCA
[U,D,V] = svd(Y_c);

figure(1)
Z_1 = U(:,1:3)'*Y_c(:,1:length(I1)); 
plot3(Z_1(1,:),Z_1(2,:),Z_1(3,:),'r.','MarkerSize',5);

hold on 
Z_2 = U(:,1:3)'*Y_c(:,length(I1)+1:length(I1)+length(I2));
plot3(Z_2(1,:),Z_2(2,:),Z_2(3,:),'b.','MarkerSize',5);

Z_3 = U(:,1:3)'*Y_c(:,length(I1)+length(I2)+1:length(II));
plot3(Z_3(1,:),Z_3(2,:),Z_3(3,:),'g.','MarkerSize',5);

set(gca,'FontSize',15)
xlabel('1st rows');
ylabel('2nd rows')
%% k-medoids


function [I_assign, I_bar] = my_k_medoids(k,D,Tau)

dQ = inf; 
count = 0;

OCT_old = 0;
OCT_new = 0;

while dQ > Tau 
    
    OCT_old = OCT_new;
    OCT_new = 0; 
      
    % Picking columns of D corresponding to current medoids 
    D_bar = D(:,k); 
    [d, I_assign] = min(D_bar'); 
    
    OCT_new = sum(d); 
    %Updating step 
    
    for i = 1:size(k)
        
        I_ell = find(I_assign == i);
        
        D_ell = D(I_ell, I_ell);
        
        
        [qi, Index] = min(sum(D_ell, 1));
        
        % We now have the new vector containing indices of 
        %characteristic vectors 
        k(1,i) = Index;
        
    end 
    
     % Calculating overall tightness 
    count = count + 1;
    
    if count == 100
      
         break
         
    end
    %Change in overall tightness 
    dQ = abs(OCT_new - OCT_old);
end

% Index vector containing the characteristic vectors
I_bar = k; 
count
OCT_new
end 
