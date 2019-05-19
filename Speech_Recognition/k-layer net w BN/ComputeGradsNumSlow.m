function [grad_b, grad_W,grad_gamma,grad_beta] = ComputeGradsNumSlow(X, Y, W, b, lambda, h,gamma,beta)

grad_W = cell(1,numel(W));
grad_b = cell(1,numel(b));

grad_gamma = cell(1,numel(gamma));
grad_beta = cell(1,numel(beta));

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda,gamma,beta);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda,gamma,beta);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda,gamma,beta);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda,gamma,beta);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(gamma)
        grad_gamma{j} = zeros(size(gamma{j}));
        for i=1:numel(gamma{j})
            
            gammas_try = gamma;
            gammas_try{j}(i) = gamma{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b, lambda,gammas_try,beta);
            
            gammas_try = gamma;
            gammas_try{j}(i) = gamma{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b, lambda,gammas_try,beta);
            
            grad_gamma{j}(i) = (c2-c1) / (2*h);
        end
    end
 for j=1:length(beta)
        grad_beta{j} = zeros(size(beta{j}));
        for i=1:numel(beta{j})
            
            beta_try = beta;
            beta_try{j}(i) = beta{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b, lambda,gamma,beta_try);
            
            beta_try = beta;
            beta_try{j}(i) = beta{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b, lambda,gamma,beta_try);
            grad_beta{j}(i) = (c2-c1) / (2*h);
        end
 end