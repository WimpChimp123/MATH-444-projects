%% Q3
load HandwrittenDigitscopy.mat X I 

% Extracting those vectors containing 2, 5, and 8 
I_1 = find(I == 2); 
I_2 = find(I == 5);
I_3 = find(I == 8);
II = [I_1, I_2, I_3];
Y = X(:,II); 
Y(:, randperm(size(Y,2)));
N = 10;

 % Running SOM 
 
 [P, gam, alp, Q] = my_som(Y, N);
 
 % Data visualization 
 
 V = zeros(160,160);
 
 rt = 1;
 
 while rt < size(P,2) + 1
     
     mini = inf;
     
     for i = 1:size(Y,2)
      
        BMU = norm(P(:,rt)-Y(:,i));
        
        if BMU < mini
            
            mini = BMU;
            jt = i;
            
        end 
        
     end 
     
     j = Q(1,rt);
     k = Q(2,rt);
   
     
     V((j-1)*16+1:j*16,(k-1)*16+1:k*16) = reshape(Y(:,jt),16,16)';
     rt = rt+1;
    
 end
 
 % Plotting 
 
 imagesc(V)
 colormap(1-gray)
 %% SOM algorithm 
 
 function [P, gam, alp, Q] = my_som(Y,N)
 
 % Generating our lattice 

Q = [;];

for i = 1:N 
    
    for j = 1:N 
        
        Q = [Q, [i;j]];
        
    end 
    
end 

% Defining our prototypes 

c = (1/size(Y,2))*sum(Y,2);

M = [];

for i = 1:size(Q,2)
    
    inter_1 = min(c)+(max(c)-min(c)).*randn(size(Y,1),1);
    M = [M, inter_1];
    
end
 
 Tmax = 500*N^2 + 1000;
 t = 0;
 T_0 = 0.1*Tmax;
 alp_0 = 0.9;
 alp_1 = 0.01;
 gam_0 = 10/3;
 gam_1 = 0.5;
 
 % Neighourhood matrix 
 

 while t < Tmax
     
     %Extracting a random data vector and obtaining the BMU
     
     xt = Y(:,randi(size(Y,2)));
     mini = inf; 
     
     for i = 1:size(Q,2)
         
         MU = norm(M(:,i)-xt);
         
         if MU < mini
             
             mini = MU; 
             BMU = M(:,i);
             jt = i; 
             
         end 
         
     end
     
     %Computing the current learning rate and coupling constant
     
     alp = max(alp_0*(1-(t/T_0)),alp_1);
     
     gam = max(gam_0*(1-(t/T_0)),gam_1);
     
     h_jt = zeros(256,1);         
    for i = 1:size(Q,2)
    
        d_ij = norm(Q(:,i)-Q(:,jt));
            
        h_jt(i) = exp((-1/2*gam^2)*d_ij^2); 
         
    end
    
    % Updating prototypes 
    for i = 1:size(Q,2)
        
        M(:,i) = M(:,i) + alp*h_jt(i)*(xt-M(:,i));
        
    end
     
    t = t + 1; 
     
 end
 
 P = M;

 end 
  
 
 
 
 


