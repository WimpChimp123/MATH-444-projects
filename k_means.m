%% Running k-means 

load WineData.mat X

one = ones(1,47);
two = ones(1,62) + 1;
three = ones(1,69) + 2;
P = [one, two, three];

[count, I_assign] = my_kmeans(P,X);

%Visualizing clustered data using PCA

I1 = find (I_assign == 1); 
I2 = find(I_assign == 2);  
I3 = find(I_assign == 3); 
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
%% k_means 

function [count, I_assign] = my_kmeans(P,X)

% Updating the characteristic vectors 

dQ = Inf;

Tau = 0.01; 

%Initial partitioning
I_1 = find(P == 1);
I_2 = find(P == 2);
I_3 = find(P == 3); 
XX = [I_1, I_2, I_3]; 

Y = X(:,XX);

%Partioning the data
Inter_1 = Y(:,1:length(I_1));

Inter_2 = Y(:,length(I_1)+1:length(I_2)+length(I_1));

Inter_3 = Y(:,length(I_2)+length(I_1)+1:length(XX));


%To store the newly partitioned data 

OCT_old = 0;
OCT_new = 0;
I_assign = zeros(1,size(X,2));
count = 0; 

while dQ > Tau 
    
    I_assign = zeros(1,size(X,2));
    OCT_old = OCT_new;
    OCT_new = 0;
    
    Index_1 = [];
    Index_2 = [];
    Index_3 = [];

    
    %Updating characteristic vectors for each partition 
    
    C1 = (1/size(Inter_1,2))*sum(Inter_1,2);
 
    C2 = (1/size(Inter_2,2))*sum(Inter_2,2);
       
    C3 = (1/size(Inter_3,2))*sum(Inter_3,2);
    
    
    
    %Within cluster tightness
    for j = 1:size(Y,2)
        tight_1 = norm(Y(:,j)-C1)^2;
        tight_2 = norm(Y(:,j)-C2)^2;
        tight_3 = norm(Y(:,j)-C3)^2;
        
        tight = [tight_1, tight_2, tight_3]; 
        [OCT_it, I] = min(tight);
        %Overall cluster tightness
        OCT_new = OCT_new + OCT_it;
        %Reassigning data vectors to the closest characteristic vector 
        if I == 1
            Index_1 = [Index_1 j];
            I_assign(1,j) = 1;
        elseif I == 2
            Index_2 = [Index_2 j];
            I_assign(1,j) = 2;
        else 
            Index_3 = [Index_3 j];
            I_assign(1,j) = 3;
        end 
        
    end
    
    %Change in overall tightness
    dQ = abs(OCT_new - OCT_old);
    
    % Updating partioning and counter
    Inter_1 = Y(:,Index_1); 
    Inter_2 = Y(:,Index_2); 
    Inter_3 = Y(:,Index_3);
    
    count = count + 1;
    
    if count == 100
       
        break
    end  
    
end
OCT_new
end 

