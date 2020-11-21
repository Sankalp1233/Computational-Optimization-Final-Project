function [w,b,t,out] = ALM_SVM_quadPenalty(X,y,lam,opts)
%=============================================
%
% augmented Lagrangian method for solving SVM
% min_{w,b,t} sum(t) + lam/2*norm(w)^2
% s.t. y(i)*(w'*X(:,i)+b) >= 1-t(i)
%      t(i) >= 0, i = 1,...,N
%
%===============================================
%
% ==============================================
% input:
%       X: training data, each column is a sample data
%       y: label vector
%       lam: model parameter
%       opts.tol: stopping tolerance
%       opts.maxit: maximum number of outer iteration
%       opts.subtol: stopping tolerance for inner-loop
%       opts.maxsubit: maxinum number of iteration for inner-loop
%       opts.w0: initial w
%       opts.b0: initial b0
%       opts.t0: initial t0
%       opts.beta: penalty parameter
%
% output:
%       w: learned w
%       b: learned b
%       out.hist_pres: historical primal residual
%       out.hist_dres: historical dual residual
%       out.hist_subit: historical iteration number of inner-loop

% ======================================================

%% get size of problem: p is dimension; N is number of data pts
[p,N] = size(X);

%% set parameters
if isfield(opts,'tol')        tol = opts.tol;           else tol = 1e-4;       end
if isfield(opts,'maxit')      maxit = opts.maxit;       else maxit = 500;      end
if isfield(opts,'subtol')     subtol = opts.subtol;     else subtol = 1e-4;    end
if isfield(opts,'maxsubit')   maxsubit = opts.maxsubit; else maxsubit = 5000;  end
if isfield(opts,'w0')         w0 = opts.w0;             else w0 = randn(p,1);  end
if isfield(opts,'b0')         b0 = opts.b0;             else b0 = 0;           end
if isfield(opts,'t0')         t0 = opts.t0;             else t0 = zeros(N,1);  end
if isfield(opts,'beta')       beta = opts.beta;         else beta = 1;         end


alpha0 = 0.5;
alpha = 0.5;
inc_ratio = 2;
dec_ratio = 0.6;

w = w0; b = b0; t = max(0,t0);
% initialize dual variable
u = zeros(N,1);

%% compute the primal residual and save to pres
s = length(t);
new_Vector= ones(s,1)-t+(-y.*(transpose(X)*w)) -y*b;
new_Vector_max = max(0,new_Vector);
pres = norm(new_Vector_max);
% save historical primal residual
hist_pres = pres;

%% compute dual residual

% compute gradient of ordinary Lagrangian function about (w,b,t)
s = length(t);
grad_w = (lam*w)-(X*(u.*y));%
grad_b = transpose(u) * -y;
grad_t = ones(s,1)-u;

id1 = t > 0;
id2 = t == 0;

% save dual residual to dres
dres =  abs(grad_b)+ norm(grad_w)... 
    + norm(grad_t(id1)) + norm(min(0,grad_t(id2)));
% 
hist_dres = dres;

hist_subit = 0;

iter = 0; subit = 0;
%% start of outer loop
while max(pres,dres) > tol && iter < maxit
    iter = iter + 1;
    % call the subroutine to update primal variable (w,b,t)
    w0 = w;
    b0 = b;
    t0 = t;
    
    % fill in the subsolver by yourself
    % if slack variables are introduced, you will have more variables
    [w,b,t] = subsolver(w0,b0,t0,subtol,maxsubit);
    
    hist_subit = [hist_subit; subit];
    
    % update multiplier u
    s = length(t);
    u = max(0, u + beta*(ones(s,1)-t+(-y.*(transpose(X)*w)) -y*b));
    
    % compute primal residual and save to hist_pres
    s = length(t);
    new_Vector= ones(s,1)-t+(-y.*(transpose(X)*w)) -y*b;
    new_Vector_max = max(0,new_Vector);
    pres = norm(new_Vector_max);
    hist_pres = [hist_pres; pres];
    
    % compute gradient of ordinary Lagrangian function about (w,b,t)
    s = length(t);
    grad_w = (lam*w)-X*(u.*y);%
    grad_b = transpose(u) * -y;
    grad_t = ones(s,1)-u;
    
    % compute the dual residual and save to hist_dres
    id1 = t > 0;
    id2 = t == 0;
    dres =  norm(grad_b)+ norm(grad_w)  ...
        + norm(grad_t(id1)) + norm(min(0,grad_t(id2)));
    %
    hist_dres = [hist_dres; dres];
    
    fprintf('out iter = %d, pres = %5.4e, dres = %5.4e, subit = %d\n',iter,pres,dres,subit);
end

out.hist_pres = hist_pres;
out.hist_dres = hist_dres;
out.hist_subit = hist_subit;

%% =====================================================
% subsolver for primal subproblem
    function [w,b,t] = subsolver(w0,b0,t0,subtol,maxsubit)
        % projected gradient for primal subproblem
        w = w0;
        b = b0;
        t = t0;
        
        % compute gradient of the augmented Lagrangian function at (w,b,t)
        s = length(t);
        Vector2= ones(s,1)-t+(-y.*(transpose(X)*w)) -y*b;
        Vector2_max = max(0,Vector2);
        grad_w = (lam*w)-X*(u.*y)- beta*((X)*(Vector2_max .*(y)));
        grad_b = ((transpose(u) * -y))-(beta*transpose(y)*Vector2_max);
        s = length(t);
        grad_t =(ones(s,1))-u-(beta*Vector2_max);
        % compute gradient error
        id1 = t > 0;
        id2 = t == 0;
%         grad_t;
%         grad_t(id1);
        grad_err =  norm(grad_b)+ norm(grad_w)...
            + norm(grad_t(id1)) + norm(min(0,grad_t(id2)) );
        % 
        s0 = 1;
        
        subit = 0;
        % start of inner-loop
        while grad_err > subtol & subit < maxsubit
            
            % compute gradient of augmented Lagrangian function at
            % (w0,b0,t0)
            s = length(t0);
            Vector3= ones(s,1)-t0+(-y.*(transpose(X)*w0)) -y*b0;
            Vector3_max = max(0,Vector3);
            grad0_w = (lam*w0)-(X*(u.*y))- beta*((X)*(Vector3_max.*(y)));%;
            grad0_b = (transpose(u) * -y)-(beta*transpose(y)*Vector3_max);
            s = length(t0);
            grad0_t = (ones(s,1))-u-(beta*Vector3_max);
            
            % evaluate the value of augmented Lagrangian function at (w0,b0,t0)
            augObj0 = transpose(t0)* ones(s,1)+(lam/2)*(norm(w0))^2 ...
                     + (transpose(u))*Vector3 + (beta/2) * (norm(Vector3_max))^2;
            augObj = inf;
            alpha = alpha0*inc_ratio;
            subit = subit + 1;
            
            % perform line search by checking local Lip continuity
            while augObj > augObj0 + (w-w0)'*grad0_w + 0.5/alpha*norm(w-w0)^2 ...
                    + (b-b0)'*grad0_b + 0.5/alpha*norm(b-b0)^2 ...
                    + (t-t0)'*grad0_t + 0.5/alpha*norm(t-t0)^2
                
                alpha = alpha*dec_ratio;
                
                % update (w,b,t) from (w0,b0,t0) by using step size alpha
                w = w0 -(alpha*grad0_w);
                b = b0 - (alpha*grad0_b);
                t = max(0,t0 - (alpha*grad0_t));
                
                % evaluate the value of augmented Lagrangian function at (w,b,t)
                s = length(t);
                Vector4= ones(s,1)-t+(-y.*(transpose(X)*w)) -y*b;
                Vector4_max = max(0,Vector4);
                augObj = transpose(t)* ones(s,1)+(lam/2)*(norm(w))^2 ...
                        + (transpose(u))* Vector4+ (beta/2) * (norm(Vector4_max))^2;
            end
            alpha0 = alpha;
            
            
            w0 = w; b0 = b; t0 = t; 
            
            % compute gradient of the augmented Lagrangian function at (w,b,t)
            s = length(t);
            Vector5= ones(s,1)-t+(-y.*(transpose(X)*w)) -y*b;
            Vector5_max = max(0,Vector5);
            grad_w = (lam*w)-(X*(u.*y))- beta*((X)*(Vector5_max.*(y)));%;
            grad_b = (transpose(u) * -y)-(beta*transpose(y)*Vector5_max);
            grad_t = (ones(s,1))-u-(beta*Vector5_max);
            % compute grad_err
            id1 = t > 0;
            id2 = t == 0;% 
            grad_err =  norm(grad_b)+norm(grad_w) ...      
                + norm(grad_t(id1)) + norm( min(0,grad_t(id2)) );
        end
    end
%=====================================================

end



