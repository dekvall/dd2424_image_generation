function J = ComputeCost(X, Y, W, b,lambda,gamma,beta,varargin)
    k=size(W,2);
    if numel(varargin)==2
        [P,~,~,~,~]=EvaluateClassifier(X,W,b,gamma,beta,varargin{1},varargin{2});
    else
        [P,~,~,~,~]=EvaluateClassifier(X,W,b,gamma,beta);
    end
    n=size(X,2);
    py= Y'*P; %Y is one hot representation of labels
    l=-log(py);
    L2=0;
    for i=1:k
        L2=L2+sumsqr(W{i});
    end
    regularization= lambda*L2;
    J= sum(diag(l))/n;
    J = J + regularization;
end