% LVQ classifier

load ECG_train.mat X_train_normal X_train_abnormal
load ECG_test.mat X_test_normal X_test_abnormal

ECG_data = [X_train_normal', X_train_abnormal'];
normal = X_train_normal';
abnormal = X_train_abnormal';

% Defining our prototypes 

c_norm = (1/size(normal,2))*sum(normal,2);
c_abnorm = (1/size(abnormal,2))*sum(abnormal,2);

m_norm = [];
m_abnorm = [];

for i = 1:5
    
    inter_1 = c_norm + abs((min(c_norm)/max(c_norm)))*randn(size(normal,1),1);
    m_norm = [m_norm, inter_1];
    
    inter_2 = c_abnorm + abs((min(c_abnorm)/max(c_abnorm)))*randn(size(abnormal,1),1);
    m_abnorm = [m_abnorm, inter_2];
    
end

M = [m_abnorm, m_norm]; 

% Learning algorithm

alp_0 = 0.9;
T = 1000;
B = log(10)/T;
updated_prototypes = [];
t = 0;

while t <= T
   
    %Extracting a random data vector and obtaining the BMU
     
     xt = ECG_data(:,randi(size(ECG_data,2)));
     mini = inf; 
     
     for i = 1:size(M,2)
         
         MU = norm(M(:,i)-xt);
         
         if MU < mini
             
             mini = MU; 
             BMU = M(:,i); 
             
         end 
         
     end
     
     %Computing the current learning rate
     
     alp = alp_0*exp(-B*t);
    
     % Updating the BMU and including the updated BMU in 
     % the m_norm and m_abnorm
     
     if find(BMU==m_norm)
         
         if find(xt==normal)
             
             BMU_new = BMU+alp*(xt-BMU);
             m_norm = [m_norm, BMU_new];
             
         else 
             
             BMU_new = BMU-alp*(xt-BMU);
             m_norm = [m_norm, BMU_new];
         end 
         
        
        
     elseif find(BMU==m_abnorm)
         
         if find(xt==abnormal)
             
             BMU = BMU+alp*(xt-BMU);
             
         else 
             
             BMU = BMU-alp*(xt-BMU);
             
         end 
         
     end
     
     updated_prototypes = [updated_prototypes, BMU];
     
     t = t+1;
    
end

% Classifying the data 

normal_heart = [];
abnormal_heart = [];
mini = inf;

for i = 1:size(ECG_data,2)
    
    x = ECG_data(:,i);
    
    for j = 1:size(updated_prototypes,2) 
        
        MU = norm(updated_prototypes(:,j)-x);
         
        if MU < mini
             
            mini = MU; 
            BMU = updated_prototypes(:,j); 
             
        end
        
    end
    
    if find(BMU==m_norm)
        
        normal_heart = [normal_heart, x];
        
    else 
        
        abnormal_heart = [abnormal_heart, x];
        
    end 
    
    mini = inf; 
    
end
     
% computing the confusion matrix and the % of vectors classified correctly

true_negative = 0;
fals_negative = 0;
 
 for j = 1:size(normal_heart,2)
     
     if find(normal_heart(:,j)==normal)
         
         true_negative = true_negative + 1;
         
     end
     
     if find(normal_heart(:,j)==abnormal)
        
         fals_negative = fals_negative + 1;
         
     end
     
 end
 
 true_positive = 0;
 fals_positive = 0;
 
 for k = 1:size(abnormal_heart,2)
     
     if find(abnormal_heart(:,k)==abnormal)
         
         true_positive = true_positive + 1;
         
     end
     
     if find(abnormal_heart(:,k)==normal)
         
         fals_positive = fals_positive + 1;
         
     end 
     
 end 
 
 % Plotting the prototypes as ECGs 
 
confusion_train = [true_negative,fals_negative;fals_positive,true_positive];

train_correct_train = 100*(true_positive+true_negative)/(size(normal,2)+size(abnormal,2))

%% Applying the classifier to our test data 

% Plotting 3 random prototypes 
 
 for j = 1:3 
     
    figure(j)
    inter = updated_prototypes(:,randi(size(updated_prototypes,2),1));
    if find(inter==m_norm)
        
        plot(inter)
        title('Normal prototype ECG')
        axis('square')
   
    else
        
        plot(inter)
        title('Abnormal prototype ECG')
        axis('square')
        
    end 
    
 end 

ECG_data_test = [X_test_normal', X_test_abnormal'];

normal_test = X_test_normal';
abnormal_test = X_test_abnormal';

normal_heart_test = [];
abnormal_heart_test = [];
mini = inf;

for i = 1:size(ECG_data_test,2)
    
    x = ECG_data_test(:,i);
    
    for j = 1:size(updated_prototypes,2) 
        
        MU = norm(updated_prototypes(:,j)-x);
         
        if MU < mini
             
            mini = MU; 
            BMU = updated_prototypes(:,j); 
             
        end
        
    end
    
    if find(BMU==m_norm)
        
        normal_heart_test = [normal_heart_test, x];
        
    else 
        
        abnormal_heart_test = [abnormal_heart_test, x];
        
    end 
    
    mini = inf; 
    
end

true_negative = 0;
fals_negative = 0;
 
 for j = 1:size(normal_heart_test,2)
     
     if find(normal_heart_test(:,j)==normal_test)
         
         true_negative = true_negative + 1;
         
     end
     
     if find(normal_heart_test(:,j)==abnormal_test)
        
         fals_negative = fals_negative + 1;
         
     end
     
 end
 
 true_positive = 0;
 fals_positive = 0;
 
 for k = 1:size(abnormal_heart_test,2)
     
     if find(abnormal_heart_test(:,k)==abnormal_test)
         
         true_positive = true_positive + 1;
         
     end
     
     if find(abnormal_heart_test(:,k)==normal_test)
         
         fals_positive = fals_positive + 1;
         
     end 
     
 end 
         
 confusion_test = [true_negative,fals_negative;fals_positive,true_positive];

test_correct = 100*(true_positive+true_negative)/(size(normal,2)+size(abnormal,2))
