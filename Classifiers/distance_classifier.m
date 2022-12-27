% Distance classifier 

load ECG_train.mat X_train_normal X_train_abnormal
load ECG_test.mat X_test_normal X_test_abnormal

% Computing the cluster means from training data

normal = X_train_normal';
abnormal = X_train_abnormal';

norm_mean = (1/size(normal,2))*sum(normal,2);
abnorm_mean = (1/size(abnormal,2))*sum(abnormal,2);

% Classifying training data using centers 

train_data = [normal, abnormal];

train_norm = [];
train_abnorm = [];

for i = 1:size(train_data,2)
    
    min_norm = norm(train_data(:,i)-norm_mean);
    min_abnorm = norm(train_data(:,i)-abnorm_mean);
    
    if min_norm < min_abnorm
        
        train_norm = [train_norm,train_data(:,i)];
        
    else
        
        train_abnorm = [train_abnorm,train_data(:,i)];
        
    end 
        
    
end 

 % computing the confusion matrix
 
 true_negative = 0;
 fals_negative = 0;
 
 for j = 1:size(train_norm,2)
     
     if find(train_norm(:,j)==normal)
         
         true_negative = true_negative + 1;
         
     end
     
     if find(train_norm(:,j)==abnormal)
        
         fals_negative = fals_negative + 1;
         
     end
     
 end
 
 true_positive = 0;
 fals_positive = 0;
 
 for k = 1:size(train_abnorm,2)
     
     if find(train_abnorm(:,k)==abnormal)
         
         true_positive = true_positive + 1;
         
     end
     
     if find(train_abnorm(:,k)==normal)
         
         fals_positive = fals_positive + 1;
         
     end 
     
 end 
         
 confusion = [true_negative,fals_negative;fals_positive,true_positive];
 
 % Percentage of vectors correctly classified 
 
 train_correct = 100*(true_positive+true_negative)/(size(normal,2)+size(abnormal,2))
 
 %% Applying distance classifier to our test data
 
 
 % Classifying testing data using training data centers 
 
normal_test = X_test_normal';
abnormal_test = X_test_abnormal';

train_data = [normal_test, abnormal_test];

train_norm = [];
train_abnorm = [];

for i = 1:size(train_data,2)
    
    min_norm = norm(train_data(:,i)-norm_mean);
    min_abnorm = norm(train_data(:,i)-abnorm_mean);
    
    if min_norm < min_abnorm
        
        train_norm = [train_norm,train_data(:,i)];
        
    else
        
        train_abnorm = [train_abnorm,train_data(:,i)];
        
    end 
        
    
end 

 % computing the confusion matrix
 
 true_negative = 0;
 fals_negative = 0;
 
 for j = 1:size(train_norm,2)
     
     if find(train_norm(:,j)==normal_test)
         
         true_negative = true_negative + 1;
         
     end
     
     if find(train_norm(:,j)==abnormal_test)
        
         fals_negative = fals_negative + 1;
         
     end
     
 end
 
 true_positive = 0;
 fals_positive = 0;
 
 for k = 1:size(train_abnorm,2)
     
     if find(train_abnorm(:,k)==abnormal_test)
         
         true_positive = true_positive + 1;
         
     end
     
     if find(train_abnorm(:,k)==normal_test)
         
         fals_positive = fals_positive + 1;
         
     end 
     
 end 

 confusion_test = [true_negative,fals_negative;fals_positive,true_positive];
 
 % Percentage of vectors correctly classified 
 
 test_correct = 100*(true_positive+true_negative)/(size(normal,2)+size(abnormal,2))