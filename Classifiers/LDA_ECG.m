% LDA 

load ECG_train.mat X_train_normal X_train_abnormal
load ECG_test.mat X_test_normal X_test_abnormal

ECG_data = [X_train_normal', X_train_abnormal'];
normal = X_train_normal';
abnormal = X_train_abnormal';

% computing the cluster means for each submatrix 

c_norm = (1/size(normal,2))*sum(normal,2);
c_abnorm = (1/size(abnormal,2))*sum(abnormal,2);

% Within cluster centering and scattering matrix

Sw_norm = normal - c_norm;
Sw_abnorm = abnormal - c_abnorm;

Scatter = [Sw_norm, Sw_abnorm];

Sw = Scatter*Scatter';

% Between cluster centering and scattering matrix 

c = (1/size(ECG_data,2))*sum(ECG_data,2);

Xj = zeros(size(ECG_data,1),size(ECG_data,2));
Xj(:,1:size(normal,2)) = c_norm.*ones(size(ECG_data,1),size(normal,2));
Xj(:,size(normal,2)+1:size(ECG_data,2)) = c_abnorm.*ones(size(ECG_data,1),size(abnormal,2));

Xb = Xj - c.*ones(size(ECG_data,1),size(ECG_data,2));

Sb = Xb*Xb';

% Largest eigenvalue/vectors and Cholesky factorization

k = chol(Sw);
kswk = inv(k')*Sb*inv(k);
[V,d1] = eig(kswk);

% computing the 1st LDA direction 

q1 = inv(k)*V(:,1);

% plotting histogram of data

q1X = zeros(1,size(ECG_data,2));

for i = 1:size(ECG_data,2)
    
    q1X(i) = q1'*ECG_data(:,i);
    
end

h = histogram(q1X,2);
title("Classes in ECG training data")
xlabel("Projections of ECG with 1st LDA direction")
ylabel("frequency")
%% Applying LDA direction to testing data

ECG_data = [X_test_normal', X_test_abnormal'];

% projecting testing data with LDA direction of training data

q1X = zeros(1,size(ECG_data,2));

for i = 1:size(ECG_data,2)
    
    q1X(i) = q1'*ECG_data(:,i);
    
end

figure(2)

histogram(q1X,2)
title("Classes in ECG testing data")
xlabel("Projections of ECG testing with 1st LDA direction from training")
ylabel("frequency")

