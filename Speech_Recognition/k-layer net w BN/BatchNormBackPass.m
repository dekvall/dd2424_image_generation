function g = BatchNormBackPass(g, si, mui, vari)
    eps  = 1e-6;
    n    = size(g,2);
    one_v = ones(n,1);
    sigma1 = ((vari + eps).^(-0.5));
    sigma2 = ((vari + eps).^(-1.5));
    G1 = g.*(sigma1*one_v');
    G2 = g.*(sigma2*one_v');
    D = si - (repmat(mui,1,size(mui,2))*one_v');
    c = (G2.*D)*one_v;
    g = G1 - (1/n)*G1*one_v-(1/n)*(D.*(c*one_v'));
end