load CongressionalVoteData.mat X I

% Removing the representative with no votes
X(:,249) = []; 
I(:,249) = [];

% Determining SMC of any two data vectors 

d_sim = zeros(size(X,2),size(X,2));

for j = 1:size(X,2)
    
    for i = 1:size(X,2)
    
        Y = X(:,j);
        Z = X(:,i); 

        count_1 = 0;
        count_0 = 0;
        count_d = 0;
        for k = 1:size(Y)
            
            if (Y(k)==1&&Z(k)==1)||(Y(k)==-1&&Z(k)==-1)
                
                %Counting the number of agreements
                
                count_1 = count_1 + 1;
        
            elseif Y(k)==0 || Z(k)==0
                
                %Excluding issues only one has voted on
                
                count_0 = count_0 + 1;
            else
                % Counting disagreements 
                
                count_d = count_d + 1; 
            end 
        
        end
        
        % Determining the similarity index
        dsmc = (count_d)/(16-count_0);
        
        if dsmc == 0 
            dsmc = 1/2;
        end 
        
        %Dissimilarity definition 

        d_sim(j,i) = dsmc; 
      
        
    end
    
end

% Picking 3 random characteristic vectors
    
C_rand = [96, 59];

% 96, 59 for best clustering, they are democrat and republican
% OCT = 91.4361

% Running k_medoids 
Tau = 0.01; 

[I_assign, I_bar] = my_k_medoids(C_rand,d_sim,Tau);

% Plotting the data 

I1 = find (I_assign == 1); 
I2 = find(I_assign == 2);   

% Computing a similarity matrix 

I3 = find (I == 1); 
I4 = find(I == 0);   
II = [I3, I4];

C = [length(I1), abs(length(I1) - length(I3)); length(I2), abs(length(I4) - length(I2))];
 

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


  