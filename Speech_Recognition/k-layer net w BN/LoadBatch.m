function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X=A.mfccs;
    %X = double(X) / double(255);
    y=A.labels';
    y=cast(y,'single');
    Y=A.onehot';
    Y=cast(Y,'single');
    X=reshape(X, [], size(X,1));
    maxX=max(abs(X));
   
  %  X=X ./max(maxX);
    
     meanX = mean(X, 2);
     stdX = std(X, 0, 2);
    
    X = X - repmat(meanX, [1, size(X, 2)]);
    X = X ./ repmat(stdX, [1, size(X, 2)]);
end