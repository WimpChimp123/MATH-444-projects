load HandwrittenDigits X I 

% Extracting those vectors containing 0, 5, and 8 
I_1 = find(I == 0); 
I_2 = find(I == 5);
I_3 = find(I == 8);
II = [I_1, I_2, I_3];
Y = X(:,II); 

[U_1,D_1,V_1] = svd(Y(:,1:length(I_1))); 

% Plotting the images of the five feature vectors

for j = 1:5
    u = U_1(:,j);
    figure(1)
    subplot(1,5,j)
    imagesc(transpose(reshape(u, 16, 16)))
    colormap(1-gray);
    axis('square')
    axis('off')

end 

[U_2,D_2,V_2] = svd(Y(:,length(I_1)+1:length(I_2)+length(I_1))); 

for i = 1:5
    u = U_2(:,i);
    figure(2)
    subplot(1,5,i)
    imagesc(transpose(reshape(u, 16, 16)))
    colormap(gray);
    axis('square')
    axis('off')
end 


[U_3,D_3,V_3] = svd(Y(:,length(I_2)+length(I_1)+1:length(II))); 

for k = 1:5
    u = U_3(:,k);
    figure(3)
    subplot(1,5,k)
    imagesc(transpose(reshape(u, 16, 16)))
    colormap(gray);
    axis('square')
    axis('off')
end 

%Approximating with principle components

Inter_1 = Y(:,1:length(I_1));
Y_1 = Inter_1(:,1); % Extracting a vector that is 0
Inter_2 = Y(:,length(I_1)+1:length(I_2)+length(I_1));
Y_2 = Inter_2(:,1); % Extracting vector that is 5
Inter_3 = Y(:,length(I_2)+length(I_1)+1:length(II)); 
Y_3 = Inter_3(:,1); % Extracting vector that is 8

Z_1 = U_1'*Y_1;
Z_2 = U_2'*Y_2;
Z_3 = U_3'*Y_3;

x_app = zeros(256,1);

for i = 1:25
    sum = Z_1(i)*U_1(:,i);
    x_app = x_app + sum;
    if mod(i, 5) == 0
        figure(4)
        subplot(1,5,i/5)
        imagesc(transpose(reshape(x_app, 16, 16)))
        colormap(1-gray)
        axis('square')
        axis('off')
    end      
end 

x_app2 = zeros(256,1);
for i = 1:25
    sum = Z_2(i)*U_2(:,i);
    x_app2 = x_app2 + sum;
    if mod(i, 5) == 0
        figure(5)
        subplot(1,5,i/5)
        imagesc(transpose(reshape(x_app2, 16, 16)))
        colormap(1-gray)
        axis('square')
        axis('off')
    end      
end 

x_app3 = zeros(256,1);
for i = 1:25
    sum = Z_3(i)*U_3(:,i);
    x_app3 = x_app3 + sum;
    if mod(i, 5) == 0
        figure(6)
        subplot(1,5,i/5)
        imagesc(transpose(reshape(x_app3, 16, 16)))
        colormap(1-gray)
        axis('square')
        axis('off')
    end      
end 

%Plotting the residual for k = 25

figure(7) 
imagesc(transpose(reshape((Y_1 - x_app), 16, 16)))
colormap(1-gray)
axis('square')
axis('off')
figure(9)
imagesc(transpose(reshape((Y_2 - x_app2), 16, 16)))
colormap(1-gray)
axis('square')
axis('off')
figure(10)
imagesc(transpose(reshape((Y_3 - x_app3), 16, 16)))
colormap(1-gray)
axis('square')
axis('off')

% plotting the norms of errors as a function of k 

k = linspace(1,25,25); 

x_err = zeros(256,1);
err_norm = zeros(25,1);
for i = 1:25
    sum = Z_3(i)*U_3(:,i);
    x_err = x_err + sum;
    err = Y_3 - x_err;
    err_norm(i) = norm(err);    
end 

figure(11)
plot(k, err_norm, 'r-')
xlabel('Truncation Values from 1 to 25')
ylabel('error norm for eights')
set(gca,'FontSize',15)

