function Y= at_l2normalize_col(X)

Y = bsxfun(@rdivide, X, sqrt(sum((X.^2),1)) );
