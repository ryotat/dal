# DAL

Dual Augmented Lagrangian (DAL) algorithm for sparse/low-rank reconstruction and learning.


## Examples

### L1-regularized squared-loss regression (LASSO)

```matlab
 m = 1024; n = 4096; k = round(0.04*n); A=randn(m,n);
 w0=randsparse(n,k); bb=A*w0+0.01*randn(m,1);
 lambda=0.1*max(abs(A'*bb));
 [ww,stat]=dalsql1(zeros(n,1), A, bb, lambda);
```

### L1-regularized logistic regression

```matlab
 m = 1024; n = 4096; k = round(0.04*n); A=randn(m,n);
 w0=randsparse(n,k); yy=sign(A*w0+0.01*randn(m,1));
 lambda=0.1*max(abs(A'*yy));
 [ww,bias,stat]=dallrl1(zeros(n,1), 0, A, yy, lambda);
```

### Grouped-L1-regularized logistic regression

```matlab
 m = 1024; n = [64 64]; k = round(0.1*n(1)); A=randn(m,prod(n));
 w0=randsparse(n,k); yy=sign(A*w0(:)+0.01*randn(m,1));
 lambda=0.1*max(sqrt(sum(reshape(A'*yy/2,n).^2)));
 [ww,bias,stat]=dallrgl(zeros(n), 0, A, yy, lambda);
```

###  Trace-norm-regularized logistic regression

```matlab
 m = 2048; n = [64 64]; r = round(0.1*n(1)); A=randn(m,prod(n));
 w0=randsparse(n,'rank',r); yy=sign(A*w0(:)+0.01*randn(m,1));
 lambda=0.2*norm(reshape(A'*yy/2,n));
 [ww,bias,stat]=dallrds(zeros(n), 0, A, yy, lambda);
```

### Matrix completion
```matlab
 n = [64 64]; r = round(0.1*n(1)); m = 2*r*sum(n);
 w0=randsparse(n,'rank',r);
 ind=randperm(prod(n)); ind=ind(1:m);
 A=sparse(1:m, ind, ones(1,m), m, prod(n));
 yy=A*w0(:)+0.01*randn(m,1);
 lambda=0.1*norm(reshape(A'*yy,n));
 [ww,stat]=dalsqds(zeros(n),A,yy,lambda);
```

### LASSO with individual weights

```matlab
 m = 1024; n = 4096; k = round(0.04*n); A=randn(m,n);
 w0=randsparse(n,k); bb=A*w0+0.01*randn(m,1);
 pp=0.1*abs(A'*bb);
 [ww,stat]=dalsqal1(zeros(n,1), A, bb, pp);
```

### Individually weighted sqaured loss

```matlab
 m = 1024; n = 4096; k = round(0.04*n); A=randn(m,n);
 w0=randsparse(n,k); bb=A*w0+0.01*randn(m,1);
 lambda=0.1*max(abs(A'*bb));
 weight=(1:m)';
 [ww,stat]=dalsqwl1(zeros(n,1), A, bb, lambda, weight);
 figure, plot(bb-A*ww);
```

### Elastic-net-regularized squared-loss regression

```matlab
 m = 1024; n = 4096; k = round(0.04*n); A=randn(m,n);
 w0=randsparse(n,k); bb=A*w0+0.01*randn(m,1);
 lambda=0.1*max(abs(A'*bb));
 [ww,stat]=dalsqen(zeros(n,1), A, bb, lambda, 0.5);
```

### Sparsely-connected multivariate AR model

See `s_test_hsgl.m`
