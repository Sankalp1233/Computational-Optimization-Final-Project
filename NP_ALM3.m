function [w,b,out] = NP_ALM3(X,y,alpha,opts)%opts
%=============================================
%
% augmented Lagrangian method for solving SVM
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
%       opts.beta: penalty parameter
%
% output:
%       w: learned w
%       b: learned b
%       out.hist_pres: historical primal residual
%       out.hist_dres: historical dual residual
%       out.hist_subit: historical iteration number of inner-loop

% ======================================================

% %% get size of problem: p is dimension; N is number of data pts
[p,N] = size(X);
%% set parameters
if isfield(opts,'tol')        tol = opts.tol;           else tol = 1e-4;       end
if isfield(opts,'maxit')      maxit = opts.maxit;       else maxit = 500;      end
if isfield(opts,'subtol')     subtol = opts.subtol;     else subtol = 1e-4;    end
if isfield(opts,'maxsubit')   maxsubit = opts.maxsubit; else maxsubit = 10000;  end
if isfield(opts,'w0')         w0 = opts.w0;             else w0 = randn(p,1); end
if isfield(opts,'b0')         b0 = opts.b0;             else b0 = 0;           end
if isfield(opts,'beta')       beta = opts.beta;         else beta = 1;         end

alpha20 = 0.5;
alpha2 = 0.5;
inc_ratio = 2;
dec_ratio = 0.6;

w = w0; b = b0;
% initialize dual variable
u = 0;

%% compute the primal residual and save to pres
sign_y = sign(y);
y_pos_values=sum(sign_y(:)==1);
y_neg_values=sum(sign_y(:)==-1);
x_pos = X(:,y==1); 
y_pos = y(y ==1);
x_neg = X(:,y == -1);
[m,n]= size(x_neg);
y_neg = y(y ==-1);
% [m,n]= size(X);
vector_log = log(1+exp((-y_neg.*(transpose(x_neg)*w)) - (y_neg*b)));
log_sum = transpose(vector_log)*ones(n,1);
expression = (1/y_neg_values)*log_sum- alpha;
new_expression = max(0,expression);
% pres = (new_expression);
pres = abs(new_expression);
% pres = abs(expression);
% pres = expression;
% save historical primal residual
hist_pres = pres;

%% compute dual residual

% compute gradient of ordinary Lagrangian function about (w,b,t)
% x_pos = X(:,y==1); 
% y_pos = y(y ==1);
% x_neg = X(:,y == -1);
% y_neg = y(y ==-1);
expre= exp(-y_pos .* (transpose(x_pos)*w) -(y_pos*b));%; 
expre2 = (1+exp((-y_pos .* (transpose(x_pos)*w))-(y_pos*b)));
newexpre = expre./ expre2;
neg_expre= exp(-y_neg .* (transpose(x_neg)*w) -(y_neg*b));%; 
neg_expre2 = (1+exp((-y_neg .* (transpose(x_neg)*w))-(y_neg*b)));
newnegexpre = neg_expre./neg_expre2;
% sign_y = sign(y);
% y_pos_values=sum(sign_y(:)==1);
% y_neg_values=sum(sign_y(:)==-1);
grad_w = (-1/y_pos_values)*(x_pos*(newexpre.*y_pos)) ...
-((u*(1/y_neg_values))*(x_neg*(newnegexpre.*y_neg)));
grad_b = (-1/y_pos_values)*(transpose(y_pos)*newexpre)...
-(u*(1/y_neg_values)*(transpose(y_neg)*newnegexpre));
% save dual residual to dres
dres =  norm(grad_b)+ norm(grad_w);
hist_dres = dres;
hist_subit = 0;
iter = 0; subit = 0;
%% start of outer loop
 while max(pres,dres) > tol & iter < maxit
    iter = iter + 1;
    % call the subroutine to update primal variable (w,b)
    w0 = w;
    b0 = b;
    
    % fill in the subsolver by yourself
    % if slack variables are introduced, you will have more variables
    [w,b] = subsolver(w0,b0,subtol,maxsubit);
    
    hist_subit = [hist_subit; subit];
    
    % update multiplier u
    sign_y = sign(y);
    y_pos_values=sum(sign_y(:)==1);
    y_neg_values=sum(sign_y(:)==-1);
    x_pos = X(:,y==1); 
    y_pos = y(y ==1);
    x_neg = X(:,y == -1);
    [m,n]= size(x_neg);
    y_neg = y(y ==-1);
    vector_log2 = log(1+exp((-y_neg.*(transpose(x_neg)*w)) - (y_neg*b)));
    log_sum2 = transpose(vector_log2)*ones(n,1);
    expression2 = (1/y_neg_values)*log_sum2- alpha;
    u = max(0, u + beta*(expression2));
    % compute primal residual and save to hist_pres
    new_expression2 = max(0,expression2);
    pres = abs(new_expression2);
%     pres = expression2;
    hist_pres = [hist_pres; pres];
    
    % compute gradient of ordinary Lagrangian function about (w,b)
    expre_2nd= exp(-y_pos .* (transpose(x_pos)*w) -(y_pos*b));%; 
    expre2_2nd = (1+exp((-y_pos .* (transpose(x_pos)*w))-(y_pos*b)));
    newexpre_2nd = expre_2nd./ expre2_2nd;
    neg_expre_2nd= exp(-y_neg .* (transpose(x_neg)*w) -(y_neg*b));%; 
    neg_expre2_2nd = (1+exp((-y_neg .* (transpose(x_neg)*w))-(y_neg*b)));
    newnegexpre_2nd = neg_expre_2nd./neg_expre2_2nd;
    grad_w = (-1/y_pos_values)*(x_pos*(newexpre_2nd.*y_pos)) ...
    -((u*(1/y_neg_values))*(x_neg*(newnegexpre_2nd.*y_neg)));
    grad_b = (-1/y_pos_values)*(transpose(y_pos)*newexpre_2nd)...
    -(u*(1/y_neg_values))*(transpose(y_neg)*newnegexpre_2nd);
    % compute the dual residual and save to hist_dres
    dres =  norm(grad_b)+ norm(grad_w);
    hist_dres = [hist_dres; dres];
    fprintf('out iter = %d, pres = %5.4e, dres = %5.4e, subit = %d\n',iter,pres,dres,subit);
end

out.hist_pres = hist_pres;
out.hist_dres = hist_dres;
out.hist_subit = hist_subit;

%% =====================================================
% subsolver for primal subproblem
    function [w,b] = subsolver(w0,b0,subtol,maxsubit)
        % projected gradient for primal subproblem
        w = w0;
        b = b0;
        
        % compute gradient of the augmented Lagrangian function at (w,b)
        sign_y = sign(y);
        y_pos_values=sum(sign_y(:)==1);
        y_neg_values=sum(sign_y(:)==-1);
        x_pos = X(:,y==1); 
        y_pos = y(y ==1);
        x_neg = X(:,y == -1);
        [m,n]= size(x_neg);
        y_neg = y(y ==-1);
        vector_log3 = log(1+exp((-y_neg.*(transpose(x_neg)*w)) - (y_neg*b)));
        log_sum3 = transpose(vector_log3)*ones(n,1);
        expression3 = (1/y_neg_values)*log_sum3- alpha;
        expre_3rd = exp(-y_pos .* (transpose(x_pos)*w) -(y_pos*b));%; 
        expre2_3rd = 1+exp((-y_pos .* (transpose(x_pos)*w))-(y_pos*b));
        newexpre_3rd = expre_3rd./ expre2_3rd;
        neg_expre_3rd = exp(-y_neg .* (transpose(x_neg)*w) -(y_neg*b));%; 
        neg_expre2_3rd = (1+exp((-y_neg .* (transpose(x_neg)*w))-(y_neg*b)));
        newnegexpre_3rd = neg_expre_3rd./neg_expre2_3rd;
        grad_w = (-1/y_pos_values)*(x_pos*(newexpre_3rd.*y_pos))...
        -(u)*((1/y_neg_values)*(x_neg*(newnegexpre_3rd.*y_neg)))...
        - beta*max(0,expression3)*(1/y_neg_values)*(x_neg*(newnegexpre_3rd.*y_neg));
%         -((1/y_neg_values)*(u+beta*max(0,expression3))*(x_neg*(newnegexpre.*y_neg)));
        grad_b = (-1/y_pos_values)*transpose(y_pos)*newexpre_3rd ...
        -(u)*(1/y_neg_values)*transpose(y_neg)*newnegexpre_3rd...
        - beta*max(0,expression3)*(1/y_neg_values)*transpose(y_neg)*newnegexpre_3rd;
%         -((1/y_neg_values)*(u+beta*max(0,expression3))*((y_neg)'*newnegexpre));
        % compute gradient error
        grad_err =  norm(grad_b)+ norm(grad_w);
        s0 = 1;
        subit = 0;
        % start of inner-loop
        while grad_err > subtol & subit < maxsubit
            
            % compute gradient of augmented Lagrangian function at
            % (w0,b0)
             sign_y = sign(y);
             y_pos_values=sum(sign_y(:)==1);
             y_neg_values=sum(sign_y(:)==-1);
             x_pos = X(:,y==1); 
             [m,n]= size(x_pos);
             y_pos = y(y ==1);
             x_neg = X(:,y == -1);
             [h,a]= size(x_neg);
             y_neg = y(y ==-1);
             vector_log4 = log(1+exp((-y_neg.*(transpose(x_neg)*w0)) - (y_neg*b0)));
             log_sum4 = transpose(vector_log4)*ones(a,1);
             expression4 = (1/y_neg_values)*log_sum4-alpha;
             expre_4th = exp(-y_pos .* (transpose(x_pos)*w0) -(y_pos*b0));%; 
             expre2_4th = (1+exp((-y_pos .* (transpose(x_pos)*w0))-(y_pos*b0)));
             newexpre_4th = expre_4th./ expre2_4th;
             neg_expre_4th= exp(-y_neg .* (transpose(x_neg)*w0) -(y_neg*b0));%; 
             neg_expre2_4th = (1+exp((-y_neg .* (transpose(x_neg)*w0))-(y_neg*b0)));
             newnegexpre_4th = neg_expre_4th./neg_expre2_4th;
             grad0_w = (-1/y_pos_values)*(x_pos*(newexpre_4th.*y_pos)) ...
             -(u)*((1/y_neg_values)*(x_neg*(newnegexpre_4th.*y_neg)))...
             - beta*max(0,expression4)*(1/y_neg_values)*(x_neg*(newnegexpre_4th.*y_neg));
             grad0_b = (-1/y_pos_values)*transpose(y_pos)*newexpre_4th ...
             -(u)*((1/y_neg_values)*transpose(y_neg)*newnegexpre_4th)...
             - beta*max(0,expression4)*(1/y_neg_values)*transpose(y_neg)*newnegexpre_4th;
            vector_log5 = log(1+exp((-y_pos.*(transpose(x_pos)*w0)) - (y_pos*b0)));
            log_sum5 = transpose(vector_log5)*ones(n,1);
            expression5 = (1/y_pos_values)*log_sum5;
            augObj0 =  expression5 + u*expression4 ...
                     + (beta/2)*(max(0,expression4))^2;
            augObj = inf;
            alpha2 = alpha20*inc_ratio;
            subit = subit + 1;
            
            % perform line search by checking local Lip continuity
            while augObj > augObj0 + (w-w0)'*grad0_w + 0.5/alpha2*norm(w-w0)^2 ...
                    + (b-b0)'*grad0_b + 0.5/alpha2*norm(b-b0)^2 
                alpha2 = alpha2*dec_ratio;
                % update (w,b) from (w0,b0) by using step size alpha
                w = w0 -(alpha2*grad0_w);
                b = b0 - (alpha2*grad0_b);
                
                % evaluate the value of augmented Lagrangian function at (w,b)
                sign_y = sign(y);
                y_pos_values=sum(sign_y(:)==1);
                y_neg_values=sum(sign_y(:)==-1);
                x_pos = X(:,y==1); 
                y_pos = y(y ==1);
                x_neg = X(:,y == -1);
                y_neg = y(y ==-1);
                [m,n]= size(x_neg);
                [h,a]= size(x_pos);
                vector_log6 = log(1+exp((-y_neg.*(transpose(x_neg)*w)) - (y_neg*b)));
                log_sum6 = transpose(vector_log6)*ones(n,1);
                expression6 = (1/y_neg_values)*log_sum6- alpha;
                vector_log7 = log(1+exp((-y_pos.*(transpose(x_pos)*w)) - (y_pos*b)));
                log_sum7 = transpose(vector_log7)*ones(a,1);
                expression7 = (1/y_pos_values)*log_sum7;
                augObj = expression7 + u*expression6 ...
                     + (beta/2)*(max(0,expression6))^2;
            end
            alpha20 = alpha2;
            
            
            w0 = w; b0 = b; 
            
            % compute gradient of the augmented Lagrangian function at (w,b)
            sign_y = sign(y);
            y_pos_values=sum(sign_y(:)==1);
            y_neg_values=sum(sign_y(:)==-1);
            x_pos = X(:,y==1); 
            y_pos = y(y ==1);
            x_neg = X(:,y == -1);
            y_neg = y(y ==-1);
            [h,a]= size(x_pos);
            [m,n]= size(x_neg);
            vector_log8 = log(1+exp((-y_neg.*(transpose(x_neg)*w)) - (y_neg*b)));
            log_sum8 = transpose(vector_log8)*ones(n,1);
            expression8 = (1/y_neg_values)*log_sum8- alpha;
            expre= exp(-y_pos .* (transpose(x_pos)*w) -(y_pos*b));%; 
            expre2 = (1+exp((-y_pos .* (transpose(x_pos)*w))-(y_pos*b)));
            newexpre = expre./ expre2;
            neg_expre= exp(-y_neg .* (transpose(x_neg)*w) -(y_neg*b));%; 
            neg_expre2 = (1+exp((-y_neg .* (transpose(x_neg)*w))-(y_neg*b)));
            newnegexpre = neg_expre./neg_expre2;
            grad_w = (-1/y_pos_values)*(x_pos*(newexpre.*y_pos)) ...;
            -(u)*(1/y_neg_values)*(x_neg*(newnegexpre.*y_neg))...
            - beta*max(0,expression8)*(1/y_neg_values)*(x_neg*(newnegexpre.*y_neg));
%             -((1/y_neg_values)*(u+beta*max(0,expression8))*(x_neg*(newnegexpre.*y_neg)))
            grad_b = (-1/y_pos_values)*transpose(y_pos)*newexpre ...
             -(u)*(1/y_neg_values)*transpose(y_neg)*newnegexpre...
             - beta*max(0,expression8)*(1/y_neg_values)*transpose(y_neg)*newnegexpre;
%             -((1/y_neg_values)*(u+beta*max(0,expression8))*((y_neg)'*newnegexpre));
           grad_err =  norm(grad_b)+norm(grad_w);
        end
    end
%=====================================================

end