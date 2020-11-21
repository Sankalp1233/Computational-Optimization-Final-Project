load ranData_rho02;

% call your solver to have (w,b)
% you can tune the parameter lambda (default 0.1)
% change the parameters if needed
x_pos = Xtrain(:,ytrain==1); 
[p,N] = size(x_pos);
alpha = .1;
w_init = randn(p,1);
b_init = 0;
opts = [];
opts.tol = 1e-4;
opts.maxit = 170;
opts.subtol = 1e-4;
opts.maxsubit = 10000;
opts.beta = 1;
opts.w0 = w_init;
opts.b0 = b_init;

fprintf('Testing by student code\n\n');

t0 = tic;

[w_s,b_s,out_s] = NP_ALM3(Xtrain,ytrain,alpha,opts);
time = toc(t0);
Xtest_pos = Xtest(:,ytest==1);
ytest_pos = ytest(ytest==1);

Xtest_neg = Xtest(:,ytest==-1);
ytest_neg = ytest(ytest==-1);

pred_y_pos = sign(Xtest_pos'*w_s + b_s);
% s=sign(pred_y_pos);
% ipositif=sum(s(:)==1);
% inegatif=sum(s(:)==-1);
pred_y_neg = sign(Xtest_neg'*w_s + b_s);
nzeros=numel(pred_y_pos==ytest_pos)-nnz(pred_y_pos==ytest_pos);

accu_pos = sum(pred_y_pos==ytest_pos)/length(ytest_pos);
accu_neg = sum(pred_y_neg==ytest_neg)/length(ytest_neg);

fprintf('classification accuracy on positive testing data: %4.2f%%\n',accu_pos*100);
fprintf('classification accuracy on negative testing data: %4.2f%%\n\n',accu_neg*100);
fprintf('false negative: %4.2f%%\n',100-(accu_pos*100));
fprintf('false positive: %4.2f%%\n',100-(accu_neg*100));
fprintf('Running time is %5.4f\n',time);
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(out_s.hist_pres,'b-','linewidth',2);
hold on
semilogy(out_s.hist_dres,'r-','linewidth',2);
legend('Primal residual','dual residual','location','best');
xlabel('outer iteration');
ylabel('error');
title('ranData\_rho02');
set(gca,'fontsize',14)