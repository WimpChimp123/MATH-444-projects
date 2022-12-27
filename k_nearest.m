% k-nearest neighbor 

load ECG_train.mat X_train_normal X_train_abnormal
load ECG_test.mat X_test_normal X_test_abnormal

ECG_data = [X_train_normal', X_train_abnormal'];
normal = X_train_normal';
abnormal = X_train_abnormal';

% Computing the distance matrix 

D = zeros(size(ECG_data,2),size(ECG_data,2)); 
dd = zeros(1,size(ECG_data,2));

for i = 1:size(ECG_data,2)
    
    dd = zeros(1,size(ECG_data,2));
    
    for j = 1:size(ECG_data,2)
        dd(1,j) = norm(ECG_data(:,j) - ECG_data(:,i));
  
    end 

    D(i,:) = dd(1,:);
  
end 

% ensuring 0s have no influence 

for i = 1:size(D,2)
    
    index = find(D(:,i)==0);
    
    D(index,i) = inf;
    
end

% Classifying using k-nearest neighbours

k = 3;
norm_vote = 0;
abnorm_vote = 0;
normal_heart = [];
abnormal_heart = [];

for j = 1:size(D,2)
    
    inter = D(:,j);
    
    for i = 1:k
    
        [minim, index] = min(inter);
        data_vec = ECG_data(:,index);
        
        if find(data_vec == normal)
            
            norm_vote = norm_vote + 1;
            
        else
            
            abnorm_vote = abnorm_vote +1;
            
        end 
        
        inter(index) = [];
                
    end
    
    if norm_vote > abnorm_vote 
        
        normal_heart = [normal_heart, ECG_data(:,j)];
        
    else
        
        abnormal_heart = [abnormal_heart, ECG_data(:,j)];
        
    end 
    
    norm_vote = 0;
    abnorm_vote = 0;
    
end

% Computing the confusion matrix and % vectors correctly classified

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
 
 for i = 1:size(abnormal_heart,2)
     
     if find(abnormal_heart(:,i)==abnormal)
         
         true_positive = true_positive + 1;
         
     end
     
     if find(abnormal_heart(:,i)==normal)
         
         fals_positive = fals_positive + 1;
         
     end 
     
 end 
         
 confusion = [true_negative,fals_negative;fals_positive,true_positive];

train_correct = 100*(true_positive+true_negative)/(size(normal,2)+size(abnormal,2))

%% Applying k-nearest neighbours to testing data

% Classifying using k-nearest neighbours

ECG_data = [X_test_normal', X_test_abnormal'];
normal = X_test_normal';
abnormal = X_test_abnormal';

k = 3;
norm_vote = 0;
abnorm_vote = 0;
normal_heart = [];
abnormal_heart = [];

for j = 1:size(D,2)
    
    inter = D(:,j);
    
    for i = 1:k
    
        [minim, index] = min(inter);
        data_vec = ECG_data(:,index);
        
        if find(data_vec == normal)
            
            norm_vote = norm_vote + 1;
            
        else
            
            abnorm_vote = abnorm_vote +1;
            
        end 
        
        inter(index) = [];
                
    end
    
    if norm_vote > abnorm_vote 
        
        normal_heart = [normal_heart, ECG_data(:,j)];
        
    else
        
        abnormal_heart = [abnormal_heart, ECG_data(:,j)];
        
    end 
    
    norm_vote = 0;
    abnorm_vote = 0;
    
end

% Computing the confusion matrix and % vectors correctly classified

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
 
 for i = 1:size(abnormal_heart,2)
     
     if find(abnormal_heart(:,i)==abnormal)
         
         true_positive = true_positive + 1;
         
     end
     
     if find(abnormal_heart(:,i)==normal)
         
         fals_positive = fals_positive + 1;
         
     end 
     
 end 
         
confusion_test = [true_negative,fals_negative;fals_positive,true_positive];

test_correct = 100*(true_positive+true_negative)/(size(normal,2)+size(abnormal,2))
