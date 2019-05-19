function acc = ComputeAccuracy(X, y, W, b,gamma,beta)
    [P,~,~,~,~]=EvaluateClassifier(X,W,b,gamma,beta);
    [~,I]= max(P,[],1);
    correct=0;
    img=size(I,2);
    for i=1:img
        if I(i)==y(i) %y are the true labels
            correct=correct + 1;
        end
    end
  acc= correct/img;
end