%% classification on ranData_rho02

load ranData_rho02;

% call your solver to have (w,b)
% you can tune the parameter lambda (default 0.1)
% change the parameters if needed

[p,N] = size(Xtrain);

lam = 1000;
w_init = randn(p,1);
b_init = 0;
t_init = randn(N,1);

opts = [];
opts.tol = 1e-1;
opts.maxit = 1000;
opts.subtol = 1e-4;
opts.maxsubit = 10000;
opts.beta = 1;
opts.w0 = w_init;
opts.b0 = b_init;
opts.t0 = t_init;

fprintf('Testing by student code\n\n');


t0 = tic;

[w_s,b_s,t_s,out_s] = ALM_SVM_quadPenalty(Xtrain,ytrain,lam,opts);

time = toc(t0);

pred_y = sign(Xtest'*w_s + b_s);

accu = sum(pred_y==ytest)/length(ytest);

fprintf('Running time is %5.4f\n',time);
fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(out_s.hist_pres,'b-','linewidth',2);
hold on
semilogy(out_s.hist_dres,'r-','linewidth',2);
legend('Primal residual','dual residual','location','best');
xlabel('outer iteration');
ylabel('error');
title('ranData\_rho02');
set(gca,'fontsize',14)



fprintf('Testing by instructor code\n\n');

lam =1;
opts = [];
opts.tol = 1e-1;
opts.maxit = 1000;
opts.subtol = 1e-4;
opts.maxsubit = 10000;
opts.beta = 1;
opts.w0 = w_init;
opts.b0 = b_init;
opts.t0 = t_init;




t0 = tic;
[w_p,b_p,out_p] = ALM_SVM_p(Xtrain,ytrain,lam,opts);
time = toc(t0);

% do classification on the testing data

pred_y = sign(Xtest'*w_p + b_p);

accu = sum(pred_y==ytest)/length(ytest);

fprintf('Running time is %5.4f\n',time);
fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(out_p.hist_pres,'b-','linewidth',2);
hold on
semilogy(out_p.hist_dres,'r-','linewidth',2);
legend('Primal residual','dual residual','location','best');
xlabel('outer iteration');
ylabel('error');
title('ranData\_rho02');
set(gca,'fontsize',14)