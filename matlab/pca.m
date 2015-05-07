function varargout = pca(A,k,its,l)
%PCA  Low-rank approximation in SVD form.
%
%
%   [U,S,V] = PCA(A)  constructs a nearly optimal rank-6 approximation
%             USV' to A, using 2 full iterations of a block Lanczos method
%             of block size 6+2=8, started with an n x 8 random matrix,
%             when A is m x n; the ref. below explains "nearly optimal."
%             The smallest dimension of A must be >= 6 when A is
%             the only input to PCA.
%
%   [U,S,V] = PCA(A,k)  constructs a nearly optimal rank-k approximation
%             USV' to A, using 2 full iterations of a block Lanczos method
%             of block size k+2, started with an n x (k+2) random matrix,
%             when A is m x n; the ref. below explains "nearly optimal."
%             k must be a positive integer <= the smallest dimension of A.
%
%   [U,S,V] = PCA(A,k,its)  constructs a nearly optimal rank-k approx. USV'
%             to A, using its full iterations of a block Lanczos method
%             of block size k+2, started with an n x (k+2) random matrix,
%             when A is m x n; the ref. below explains "nearly optimal."
%             k must be a positive integer <= the smallest dimension of A,
%             and its must be a nonnegative integer.
%
%   [U,S,V] = PCA(A,k,its,l)  constructs a nearly optimal rank-k approx.
%             USV' to A, using its full iterates of a block Lanczos method
%             of block size l, started with an n x l random matrix,
%             when A is m x n; the ref. below explains "nearly optimal."
%             k must be a positive integer <= the smallest dimension of A,
%             its must be a nonnegative integer,
%             and l must be a positive integer >= k.
%
%
%   The low-rank approximation USV' is in the form of an SVD in the sense
%   that the columns of U are orthonormal, as are the columns of V,
%   the entries of S are all nonnegative, and the only nonzero entries
%   of S appear in non-increasing order on its diagonal.
%   U is m x k, V is n x k, and S is k x k, when A is m x n.
%
%   Increasing its or l improves the accuracy of the approximation USV'
%   to A; the ref. below describes how the accuracy depends on its and l.
%
%
%   Note: PCA invokes RAND. To obtain repeatable results,
%         invoke RAND('seed',j) with a fixed integer j before invoking PCA.
%
%   Note: PCA currently requires the user to center and normalize the rows
%         or columns of the input matrix A before invoking PCA (if such
%         is desired).
%
%   Note: The user may ascertain the accuracy of the approximation USV'
%         to A by invoking DIFFSNORM(A,U,S,V).
%
%
%   inputs (the first is required):
%   A -- matrix being approximated
%   k -- rank of the approximation being constructed;
%        k must be a positive integer <= the smallest dimension of A,
%        and defaults to 6
%   its -- number of full iterations of a block Lanczos method to conduct;
%          its must be a nonnegative integer, and defaults to 2
%   l -- block size of the block Lanczos iterations;
%        l must be a positive integer >= k, and defaults to k+2
%
%   outputs (all three are required):
%   U -- m x k matrix in the rank-k approximation USV' to A,
%        where A is m x n; the columns of U are orthonormal
%   S -- k x k matrix in the rank-k approximation USV' to A,
%        where A is m x n; the entries of S are all nonnegative,
%        and its only nonzero entries appear in nonincreasing order
%        on the diagonal
%   V -- n x k matrix in the rank-k approximation USV' to A,
%        where A is m x n; the columns of V are orthonormal
%
%
%   Example:
%     A = rand(1000,2)*rand(2,1000);
%     A = A/normest(A);
%     [U,S,V] = pca(A,2,0);
%     diffsnorm(A,U,S,V)
%
%     This code snippet produces a rank-2 approximation USV' to A such that
%     the columns of U are orthonormal, as are the columns of V, and
%     the entries of S are all nonnegative and are zero off the diagonal.
%     diffsnorm(A,U,S,V) outputs an estimate of the spectral norm
%     of A-USV', which should be close to the machine precision.
%
%
%   Reference:
%   Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
%   Finding structure with randomness: Stochastic algorithms
%   for constructing approximate matrix decompositions,
%   arXiv:0909.4061 [math.NA; math.PR], 2009
%   (available at http://arxiv.org).
%
%
%   See also PCACOV, PRINCOMP, SVDS.
%

%   Copyright 2009 Mark Tygert.

%
% Check the number of inputs.
%
if(nargin < 1)
  error('MATLAB:pca:TooFewIn',...
        'There must be at least 1 input.')
end

if(nargin > 4)
  error('MATLAB:pca:TooManyIn',...
        'There must be at most 4 inputs.')
end

%
% Check the number of outputs.
%
if(nargout~=1 && nargout ~= 3)
  error('MATLAB:pca:WrongNumOut',...
        'There must be exactly 3 outputs.')
end

%
% Set the inputs k, its, and l to default values, if necessary.
%
if(nargin == 1)
  k = 6;
  its = 2;
  l = k+2;
end

if(nargin == 2)
  its = 2;
  l = k+2;
end

if(nargin == 3)
  l = k+2;
end

%
% Check the first input argument.
%
if(~isfloat(A) && ~iscell(A))
  error('MATLAB:pca:In1NotFloat',...
        'Input 1 must be a floating-point matrix.')
end

if(isempty(A))
  error('MATLAB:pca:In1Empty',...
        'Input 1 must not be empty.')
end

%
% Retrieve the dimensions of A.
%
if isnumeric(A)
  [m, n] = size(A);
else
  AT=A{2};
  m=A{3};
  n=A{4};
  A=A{1};
end

%
% Check the remaining input arguments.
%
if(size(k,1) ~= 1 || size(k,2) ~= 1)
  error('MATLAB:pca:In2Not1x1',...
        'Input 2 must be a scalar.')
end

if(size(its,1) ~= 1 || size(its,2) ~= 1)
  error('MATLAB:pca:In3Not1x1',...
        'Input 3 must be a scalar.')
end

if(size(l,1) ~= 1 || size(l,2) ~= 1)
  error('MATLAB:pca:In4Not1x1',...
        'Input 4 must be a scalar.')
end

if(k <= 0)
  error('MATLAB:pca:In2NonPos',...
        'Input 2 must be > 0.')
end

if((k > m) || (k > n))
  error('MATLAB:pca:In2TooBig',...
        'Input 2 must be <= the smallest dimension of Input 1.')
end

if(its < 0)
  error('MATLAB:pca:In3Neg',...
        'Input 3 must be >= 0.')
end

if(l < k)
  error('MATLAB:pca:In4ltIn2',...
        'Input 4 must be >= Input 2.')
end

%
% SVD A directly if (its+1)*l >= m/1.25 or (its+1)*l >= n/1.25.
%
if(((its+1)*l >= m/1.25) || ((its+1)*l >= n/1.25))

  if isnumeric(A)
  if(~issparse(A))
    if nargout>1
      [U,S,V] = svd(A,'econ');
    else
      S =  svd(A,'econ');
    end
  end

  if(issparse(A))
    if nargout>1
      [U,S,V] = svd(full(A),'econ');
    else
      S = svd(full(A),'econ');
    end
  end
  else
    if nargout>1
      [U,S,V]=svd(A(eye(n)),'econ');
    else
      S = svd(A(eye(n)),'econ');
    end
  end

elseif(m >= n)

%
% Apply A to a random matrix, obtaining H.
%
  rand('seed',rand('seed'));

  if isnumeric(A)
  if(isreal(A))
    H = A*(2*rand(n,l)-ones(n,l));
  end

  if(~isreal(A))
    H = A*( (2*rand(n,l)-ones(n,l)) + i*(2*rand(n,l)-ones(n,l)) );
  end
  else
    H = A(2*rand(n,l)-ones(n,l));
  end
  

  rand('twister',rand('twister'));

%
% Initialize F to its final size and fill its leftmost block with H.
%
  F = zeros(m,(its+1)*l);
  F(1:m, 1:l) = H;

%
% Apply A*A' to H a total of its times,
% augmenting F with the new H each time.
%
  for it = 1:its
    if isnumeric(A)
      H = (H'*A)';
      H = A*H;
    else
      H = A(AT(H));
    end
    F(1:m, (1+it*l):((it+1)*l)) = H;
  end

  clear H;

%
% Form a matrix Q whose columns constitute an orthonormal basis
% for the columns of F.
%
  [Q,R,E] = qr(F,0);

  clear F R E;

%
% SVD Q'*A to obtain approximations to the singular values
% and right singular vectors of A; adjust the left singular vectors
% of Q'*A to approximate the left singular vectors of A.
%
  if isnumeric(A)
    if nargout>1
      [U2,S,V] = svd(Q'*A,'econ');
    else
      S = svd(Q'*A,'econ');
    end
  else
    if nargout>1
      [V,S,U2] = svd(AT(Q),'econ');
    else
      S = svd(AT(Q),'econ');
    end
  end
  if nargout>1
    U = Q*U2;
  end

  clear Q U2;

elseif(m < n)

%
% Apply A' to a random matrix, obtaining H.
%
  rand('seed',rand('seed'));

  if isnumeric(A)
  if(isreal(A))
    H = ((2*rand(l,m)-ones(l,m))*A)';
  end

  if(~isreal(A))
    H = (( (2*rand(l,m)-ones(l,m)) + i*(2*rand(l,m)-ones(l,m)) )*A)';
  end
  else % assume A is real
    H = AT(2*rand(m,l)-ones(m,l));
  end
  
  rand('twister',rand('twister'));

%
% Initialize F to its final size and fill its leftmost block with H.
%
  F = zeros(n,(its+1)*l);
  F(1:n, 1:l) = H;

%
% Apply A'*A to H a total of its times,
% augmenting F with the new H each time.
%
  for it = 1:its
    if isnumeric(A)
    H = A*H;
    H = (H'*A)';
    else
      H=AT(A(H));
    end
    F(1:n, (1+it*l):((it+1)*l)) = H;
  end

  clear H;

%
% Form a matrix Q whose columns constitute an orthonormal basis
% for the columns of F.
%
  [Q,R,E] = qr(F,0);

  clear F R E;

%
% SVD A*Q to obtain approximations to the singular values
% and left singular vectors of A; adjust the right singular vectors
% of A*Q to approximate the right singular vectors of A.
%
  if isnumeric(A)
    if nargout>1
      [U,S,V2] = svd(A*Q,'econ');
    else
      S = svd(A*Q,'econ');
    end
  else
    if nargout>1
      [V2,S,U] = svd(A(Q),'econ');
    else
      S = svd(A(Q),'econ');
    end
  end
  if nargout>1
    V = Q*V2;
  end

  clear Q V2;


end
%
% Retain only the leftmost k columns of U, the leftmost k columns of V,
% and the uppermost leftmost k x k block of S.
%

if nargout>1
  varargout{1} = U(:,1:k);
  varargout{2} = S(1:k,1:k);
  varargout{3} = V(:,1:k);
else
  varargout{1} = S(1:k);
end
