% Computing GLCM 

load TestImages.mat X I 
 
% Gray levels of the image 

k = 8; 

for u = 1:3
    
    X_param1 = zeros(k*k,350);
    
    for v = 1:size(X,2)

        % Extracting each image 
        
        X_inter = X(:, v);
        X_inter = reshape(X_inter,128,128);


        % Dividing up the subintervals

        interj = zeros(k,2); 

        for i = 1:k-1
    
            interj(i,:) = [(i-1)/k, i/k];
        
        end 

        interj(k,:) = [(k-1)/k,1];

        % Defining the coarse image C

        C = zeros(size(X_inter,1),size(X_inter,2));
        count = 1;

        for i = 1:size(X_inter,1)
    
            for j = 1:size(X_inter,2)
            
                idx = find(X_inter(i,j)>=interj(:,1) & X_inter(i,j)<=interj(:,2));
        
                C(i,j) = idx;
        
            end
  
        end

        % Computing three GLCMs and normalizing for each image
        
        n_1 = size(C,1);
        n_2 = size(C,2);
    
        % GLCMS for three different parameters
        
        if u == 1
            
            mu = 1;
            nu = 2;
            C_shift= NaN(n_1,n_2);
            C_shift(1:n_1-mu,1:n_2-nu) = C(1+mu:n_1,1+nu:n_2);
            
        elseif u == 2
            
            mu = 4;
            nu = 0;
            C_shift= NaN(n_1,n_2);
            C_shift(1:n_1-mu,1:n_2-nu) = C(1+mu:n_1,1+nu:n_2);
            
        else 
            mu = 3;
            nu = 2;
            C_shift= NaN(n_1,n_2);
            C_shift(1:n_1-mu,1:n_2-nu) = C(1+mu:n_1,1+nu:n_2);
            
        end
    
            G = zeros(k,k);

            for i = 1:k
    
                Ii = (C == i);
    
                for j = 1:k
        
                    Ij = (C_shift == j);
        
                    G(i,j) = sum(sum(Ii.* Ij));
        
                end
    
            end
            
        G = reshape(G,k*k,1);
        
        % Normalizing the GLCM
        
        N = norm(G,1);
        X_param1(:,v) = (1/N).*G;
    
    end 
    
    % Collecting the 3 GLCMS
    if u == 1
        
        X_02 = X_param1;
        
    elseif u == 2
        
        X_20 = X_param1;
        
    else 
        
        X_22 = X_param1;
        
    end 

end

% Computing the reduced image data

X_reduced = [X_02;X_20;X_22];

% Further reduction of dimensionality using PCA

[U,D,V] = svd(X_reduced);
Z_reduced = U(:,1:4)'*X_reduced;

%% 1&2 PCA

% Ordering X_reduced for plotting

I_1 = find(I==1);
I_2 = find(I==2);
I_3 = find(I==3);
I_4 = find(I==4);
I_5 = find(I==5);
II = [I_1, I_2, I_3, I_4, I_5];
X_ord = X_reduced(:, II);

% Plotting pairs of principle components

for i = 1:size(X_ord,2)
    
    z = U(:,1:4)'*X_ord(:,i);
    
    if (i>=1)&&(i<= 70)
        
        plot(z(1), z(2), 'r.');
        hold on 
        
    elseif (70<i)&&(i<=140)
    
        plot(z(1), z(2), 'b.');
        hold on
        
    elseif (140<i)&&(i<= 210)
        
        plot(z(1), z(2), 'g.')
        hold on
        
    elseif (210<i)&&(i<=280)
        
        plot(z(1), z(2), 'k.')
        hold on
        
    elseif (280<i)&&(i<=350)
        
        plot(z(1), z(2), 'y.')
        hold on
        
    end
    
    z = zeros(4,1);
    
end

title('Principle components Z1 and Z2')

%% LDA 

% Assign data into submatrices and computing the center

I_1 = find(I==1);
I_2 = find(I==2);
I_3 = find(I==3);
I_4 = find(I==4);
I_5 = find(I==5);
II = [I_1, I_2, I_3, I_4, I_5];
Z_ord = Z_reduced(:, II);
c = (1/size(Z_ord,2))*sum(Z_ord,2);

% computing the cluster means for each submatrix 

CY_1 = (1/70)*sum(Z_ord(:,1:70),2);
CY_2 = (1/70)*sum(Z_ord(:,71:140),2);
CY_3 = (1/70)*sum(Z_ord(:,141:210),2);
CY_4 = (1/70)*sum(Z_ord(:,211:280),2);
CY_5 = (1/70)*sum(Z_ord(:,281:350),2);

% Within cluster centering and scattering matrix

SX_1 = Z_ord(:,1:70) - CY_1;
SX_2 = Z_ord(:,71:140) - CY_2;
SX_3 = Z_ord(:,141:210) - CY_3;
SX_4 = Z_ord(:,211:280) - CY_4;
SX_5 = Z_ord(:,281:350) - CY_5;

Xw = [SX_1, SX_2, SX_3, SX_4, SX_5];

Sw = Xw*Xw';

% Between cluster centering and scattering matrix 

Xj = zeros(size(Z_ord,1),size(Z_ord,2));
Xj(:,1:70) = CY_1.*ones(4,70);
Xj(:,71:140) = CY_2.*ones(4,70);
Xj(:,141:210) = CY_3.*ones(4,70);
Xj(:,211:280) = CY_4.*ones(4,70);
Xj(:,281:350) = CY_5.*ones(4,70);

Xb = Xj - c.*ones(size(Z_ord,1),size(Z_ord,2));

Sb = Xb*Xb';

% Largest eigenvalue/vectors and Cholesky factorization

k = chol(Sw);
kswk = inv(k')*Sb*inv(k);
[V,d1] = eig(kswk);

% computing the 1st three LDA directions 

q = inv(k)*V(:,1:3);

% Checking the separation by plotting pairs of LDA projections

for i = 1:size(Z_ord,2)
    
    z = q'*Z_ord(:,i);
    
    if (i>=1)&&(i<= 70)
        
        plot3(z(1), z(2), z(3), 'r.');
        hold on 
        
    elseif (70<i)&&(i<=140)
    
        plot3(z(1), z(2), z(3), 'b.');
        hold on
        
    elseif (140<i)&&(i<= 210)
        
        plot3(z(1), z(2), z(3), 'g.')
        hold on
        
    elseif (210<i)&&(i<=280)
        
        plot3(z(1), z(2), z(3), 'k.')
        hold on
        
    elseif (280<i)&&(i<=350)
        
        plot3(z(1), z(2), z(3), 'y.')
        hold on
        
    end
    
    z = zeros(4,1);
    
end

xlabel('q1')
ylabel('q2')
zlabel('q3')
title('q1, q2, q3 projections')

