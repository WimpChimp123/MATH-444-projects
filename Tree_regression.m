% Regression tree analysis 


%% Toy data

x = [0.25, 0.8, 0.6]';
y = [0.25, 0.1, 0.8]';
v = [0.1,0.5,0.6;0.4,0.1,0.3;0.9,0.2,0.0];

% Defining the root

T(1).s = [];
T(1).j = [];
T(1).left  = [];
T(1).right = [];
T(1).I     = (1:3);
T(1).x     = [0;1];
T(1).y     = [0;1];
T(1).v   = 1/size(T(1).I,2)*sum(v);

% finding optimal split values 

[j_opt, s_opt] = OptimalSplitRegression(x,y,v,T(1).I);
T(1).j = j_opt;
T(1).s = s_opt;
T(1).left = find(x<T(1).j)';
T(1).right = find(x>T(1).j)';

leaves = [1];
count = 1;
i = 1;

while 1 ~= 0
    
    % Checking if leaf is pure and splitting accordingly
    
    if ~(size(T(i).I,2)==1)
        
        leaves = leaves(leaves~=i);
        T(i+count).I = T(i).left;
        T(i+count+1).I = T(i).right;
        
        % Running optimal split
        
        [T(i+count).j, T(i+count).s] = OptimalSplitRegression(x,y,v,T(i+count).I);
        [T(i+count+1).j, T(i+count+1).s] = OptimalSplitRegression(x,y,v,T(i+count+1).I);
        
        % Sorting left and right sides
        
        if T(i+count).s == 1
            
            lind_i = find(x(T(i+count).I)<T(i+count).j);
            T(i+count).left = T(i).left(lind_i);
            rind_i = find(x(T(i+count).I)>T(i+count).j);
            T(i+count).right = T(i).left(rind_i);
            
        elseif T(i+count).s == 2
            
            lind_i = find(y(T(i+count).I)<T(i+count).j);
            T(i+count).left = T(i).left(lind_i);
            rind_i = find(y(T(i+count).I)>T(i+count).j);
            T(i+count).right = T(i).left(rind_i);
            
        end
        
        if T(i+count+1).s == 1
            
            lind_i = find(x(T(i+count+1).I)<T(i+count+1).j);
            T(i+count+1).left = T(i).right(lind_i);
            rind_i = find(x(T(i+count+1).I)>T(i+count+1).j);
            T(i+count+1).right = T(i).right(rind_i);
            
        elseif T(i+count+1).s == 2
            
            lind_i = find(y(T(i+count+1).I)<T(i+count+1).j);
            T(i+count+1).left = T(i).right(lind_i);
            rind_i = find(y(T(i+count+1).I)>T(i+count+1).j);
            T(i+count+1).right = T(i).right(rind_i);
           
        end
        
        % Filling in the remaining entries 
        
        if T(i).s == 1
            
            T(i+count).x = [T(i).x(1), T(i).j];
            T(i+count).y = T(i).y;
            T(i+count+1).x = [T(i).j, T(i).x(2)];
            T(i+count+1).y = T(i).y;
            
        else 
            
            T(i+count).x = T(i).x;
            T(i+count).y = [T(i).y(1), T(i).j];
            T(i+count+1).x = T(i).x;
            T(i+count+1).y = [T(i).j, T(i).y(2)];
            
            
        end 
        
        left_ndata = size(T(i+count).I,1);
        right_ndata = size(T(i+count+1).I,1);
        
        if size(v(T(i+count).I,:),1) == 1
            
            T(i+count).v = v(T(i+count).I,:);
            
        else 
            
            T(i+count).v = 1/left_ndata*sum(v(T(i+count).I,:));
            
        end 
        
        if size(v(T(i+count+1).I,:),1) == 1
            
            T(i+count+1).v = v(T(i+count+1).I,:);
            
        else 
            
            T(i+count+1).v = 1/right_ndata*sum(v(T(i+count+1).I,:));
            
        end 
        
        % appending the new leaves
        
        leaves = [leaves, i+count]; 
        leaves = [leaves, i+count+1];
        
        count = count + 1;
        
    else 
        
        T(i).s = [];
        T(i).j = [];
        T(i).left  = [];
        T(i).right = [];
        
    end
    
    i = i + 1;
    
    if size(leaves,2) >= 3
        
        break 
        
    end 
    
end 

%% Running with 
load MysteryImage.mat cols rows vals

m = 1456;
n = 2592; 

x_data = cols/n;
y_data = (m/n)*(1-rows/m);

% Applying the algorithm with image data 

x = x_data;
y = y_data;
v = vals;

% defining the root 

n_data = size(x_data,1);

R(1).s = [];
R(1).j = [];
R(1).left  = [];
R(1).right = [];
R(1).I     = (1:n_data);
R(1).x     = [0;1];
R(1).y     = [0;m/n];
R(1).v   = 1/n_data*sum(v);

% finding optimal split values 

[j_opt, s_opt] = OptimalSplitRegression(x_data,y_data,vals,R(1).I);
R(1).j = j_opt;
R(1).s = s_opt;
R(1).left = find(x<R(1).j);
R(1).right = find(x>R(1).j);

leaves = [1];
count = 1;
i = 1;

while 1 ~= 0
    
    % Checking if rectangles contain pixels of same RGB
    
    if ~((sum(ismember(v(R(i).I,:), v(R(i).I(1),:), 'rows')))==(size(v(R(i).I,:),1)))
        
        leaves = leaves(leaves~=i);
        
        % Filling the structure 
        
        R(i+count).I = R(i).left;
        R(i+count+1).I = R(i).right;
        
        % Computing the splitting values and adding them to structure
        
        [R(i+count).j, R(i+count).s] = OptimalSplitRegression(x,y,v,R(i).left);
        [R(i+count+1).j, R(i+count+1).s] = OptimalSplitRegression(x,y,v,R(i).right);
        
        % Finding the left and right indices for the newly computed split
        
        if R(i+count).s == 1
            
            lind_i = find(x(R(i).left)<=R(i+count).j);
            R(i+count).left = R(i).left(lind_i);
            rind_i = find(x(R(i).left)>R(i+count).j);
            R(i+count).right = R(i).left(rind_i);
            
        elseif R(i+count).s == 2
            
            lind_i = find(y(R(i).left)<=R(i+count).j);
            R(i+count).left = R(i).left(lind_i);
            rind_i = find(y(R(i).left)>R(i+count).j);
            R(i+count).right = R(i).left(rind_i);
            
        end
        
        
        if R(i+count+1).s == 1
            
            lind_i = find(x(R(i).right)<=R(i+count+1).j);
            R(i+count+1).left = R(i).right(lind_i);
            rind_i = find(x(R(i).right)>R(i+count+1).j);
            R(i+count+1).right = R(i).right(rind_i);
            
        elseif R(i+count+1).s == 2
            
            lind_i = find(y(R(i).right)<=R(i+count+1).j);
            R(i+count+1).left = R(i).right(lind_i);
            rind_i = find(y(R(i).right)>R(i+count+1).j);
            R(i+count+1).right = R(i).right(rind_i);
           
        end
        
        % Filling in the x and y bounds for the new rectangles
        
        if R(i).s == 1
            
            R(i+count).x = [R(i).x(1), R(i).j];
            R(i+count).y = R(i).y;
            R(i+count+1).x = [R(i).j, R(i).x(2)];
            R(i+count+1).y = R(i).y;
            
        else 
            
            R(i+count).x = R(i).x;
            R(i+count).y = [R(i).y(1), R(i).j];
            R(i+count+1).x = R(i).x;
            R(i+count+1).y = [R(i).j, R(i).y(2)];
            
            
        end 
        
        % Computing the average RGB for each rectangle
        
        left_ndata = size(R(i+count).I,1);
        right_ndata = size(R(i+count+1).I,1);
        
        R(i+count).v = 1/left_ndata*sum(v(R(i+count).I,:));
        R(i+count+1).v = 1/right_ndata*sum(v(R(i+count+1).I,:));
        
        % Appending the new leaves and incrementing the counter
        
        leaves = [leaves, i+count]; 
        leaves = [leaves, i+count+1];
        
        count = count + 1;
        
    else 
        
        R(i).s = [];
        R(i).j = [];
        R(i).left  = [];
        R(i).right = [];
        
        count = count - 1;
        
    end 
        
    % We want to stop at a given number of leaves 
    
    if size(leaves,2) >= 500
        
        break 
        
    end
    
    i = i + 1;
    
end 

% Plotting the image 

figure(1)

for i = 1:size(leaves,2)
    
    hold on
    
    k = leaves(i);
    
    rgb_vec = R(k).v;
    
    fill([R(k).x(1), R(k).x(2), R(k).x(2), R(k).x(1)],[R(k).y(1), R(k).y(1), R(k).y(2), R(k).y(2)],rgb_vec);
    
end 


%% optimal split algorhitm  

function [j_opt, s_opt] = OptimalSplitRegression(x_data,y_data,vals,I_curr)

% This function computes the optimal split value and direction 

% optimal x-value

Fx = inf;
leastq = 0;
x_data = x_data(I_curr);
x_data = sort(x_data, 'ascend');

y_data = y_data(I_curr);
y_data = sort(y_data, 'ascend');

if size(x_data) == 1
    
    j_opt = [];
    s_opt = [];
    
else 
    
    for i = 1:size(x_data,1)-1
    
        leastq = Fx;
    
        sigl = 0.5*(x_data(i)+x_data(i+1));
    
        % Computing c1 and c2 
    
        c1 = find(x_data<=sigl);
        c2 = find(x_data>sigl);
    
        c1_val = 1/size(c1,1)*(sum(vals(c1,:)));
        c2_val = 1/size(c2,1)*(sum(vals(c2,:)));
    
        % Computing Fx1
        
       
        summa = vals(c1,:)-c1_val;
        c1_summa = summa(:,1).^2+summa(:,2).^2+summa(:,3).^2;
        Fx1 = sum(c1_summa);
       
        % Computing Fx2
        
        summa = vals(c2,:)-c2_val;
        c2_summa = summa(:,1).^2+summa(:,2).^2+summa(:,3).^2; 
        Fx2 = sum(c2_summa);
            
        
        
        Fx = Fx1 + Fx2;
    
        % Saving the minimum
    
        if Fx < leastq 
        
            x_opt = sigl;
            x_least = Fx;
        
        end 

    end

    % optimal y-value 

    Fy = inf;
    leastq = 0;

    for i = 1:size(y_data,1)-1
    
        leastq = Fy;
    
        sigl = 0.5*(y_data(i)+y_data(i+1));
    
        % Computing c1 and c2 
    
        c1 = find(y_data<=sigl);
        c2 = find(y_data>sigl);
    
        c1_val = 1/size(c1,1)*(sum(vals(c1,:)));
        c2_val = 1/size(c2,1)*(sum(vals(c2,:)));
        
        
        % Computing Fy1
        
       
        summa = vals(c1,:)-c1_val;
        c1_summa = summa(:,1).^2+summa(:,2).^2+summa(:,3).^2;  
        Fy1 = sum(c1_summa);
       
        % Computing Fy2
        
        summa = vals(c2,:)-c2_val;
        c2_summa = summa(:,1).^2+summa(:,2).^2+summa(:,3).^2;   
        Fy2 = sum(c2_summa);
        
        Fy = Fy1 + Fy2;
    
        % Saving the minimum
    
        if Fy < leastq 
        
            y_opt = sigl;
            y_least = Fy;
        
        end 

    end
   
    if y_least < x_least
    
        s_opt = 2;
        j_opt = y_opt;
    
    else 
    
        s_opt = 1;
        j_opt = x_opt;
    
    end
        
        
        
end 

    
end 

