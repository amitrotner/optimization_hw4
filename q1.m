clc; close all; clear;
%%
[X1, X2] = meshgrid(-2:.2:2, -2:.2:2);
Y = X1 .* exp(-X1.^2 - X2.^2);
figure(); surf(X1, X2, Y);

N_train = 500;
X_train = 4*rand(2,N_train)-2;
Y_train = X_train(1,:) .* exp(-X_train(1,:).^2 - X_train(2,:).^2);

N_test=200;
X_test= 4*rand(2,N_test)-2;
Y_test = X_test(1,:) .* exp(-X_test(1,:).^2 - X_test(2,:).^2);

weights = zeros([31 1]);
W1_init = randn(2,4)/sqrt(2);
weights(1:8) = W1_init(:);
W2_init = randn(4,3)/sqrt(4);
weights(9:20) = W2_init(:);
W3_init = randn(3,1)/sqrt(3);
weights(21:23) = W3_init(:);
[weights, value_arr] = BFGS(@psi_train, weights, N_train, X_train, Y_train);

figure(); semilogy(value_arr); title('Avg Error per Iteration'); xlabel('Iterations'); ylabel('Error');

[~,~,network_reconstruction] = psi_train(weights, N_test, X_test, Y_test);

f = fit([X_test(1,:)', X_test(2,:)'],network_reconstruction,'linearinterp');
figure(); plot( f, [X_test(1,:)', X_test(2,:)'], network_reconstruction);

%%
function [tot, grads, y_hats] = psi_train(weights, N, X, Y)
    W1 = reshape(weights(1:8),[2 4]);
    W2 = reshape(weights(9:20),[4 3]);
    W3 = reshape(weights(21:23),[3 1]);
    b1 = reshape(weights(24:27),[4 1]);
    b2 = reshape(weights(28:30),[3 1]);
    b3 = weights(31);
    
    tot_grads = zeros([31 1]);
    tot = 0;
    y_hats = [];
    % for every entry in the dataset
    for i = 1:N
        x = X(:,i);
        y = Y(:,i);
        u1 = W1'*x+b1;
        phi1 = tanh(u1);
        phi1_tag = diag(1-(tanh(u1).^2));
        u2 = W2'*phi1+b2;
        phi2 = tanh(u2);
        phi2_tag = diag(1-(tanh(u2).^2));
        F = W3'*phi2+b3;
        r = F - y;
        psi = r^2;
        tot = tot + psi;
        y_hats = [y_hats; F];
        
        grad_b3 = 2 * r;
        grad_W3 = phi2 * grad_b3;
        grad_b2 = phi2_tag * W3 .* grad_b3;
        grad_W2 = phi1*grad_b2';
        grad_b1 = phi1_tag * (W2 * grad_b2);
        grad_W1 = x * grad_b1';
        
           
        
        
        tot_grads(1:8) = tot_grads(1:8) + grad_W1(:);
        tot_grads(9:20) = tot_grads(9:20) + grad_W2(:);
        tot_grads(21:23) = tot_grads(21:23) + grad_W3(:);
        tot_grads(24:27) = tot_grads(24:27) + grad_b1(:);
        tot_grads(28:30) = tot_grads(28:30) + grad_b2(:);
        tot_grads(31) = tot_grads(31) + grad_b3;
    end
    tot = tot/N;
    grads = tot_grads./N;
end

function [min_x, value_array] = BFGS(target_func, weights, N, X, Y)
    iter = 0;
    w = weights;
    B = eye(31);
    alpha0 = 1;
    sigma = 0.25;
    sigma2 = 0.9;
    beta = 0.5;
    [value, grad] = target_func(w, N, X, Y);
    value_array = [value];
    epsilon = 10^-5;
    while norm(grad) >= epsilon
        d = -1 .* B * grad;
        iter = iter + 1;
        alpha = alpha0;
        c = grad'*d;
        stop_flag = false;
        counter = 1;
        while ~stop_flag
            counter = counter + 1;
            value_tmp = target_func(w + alpha.*d, N, X, Y);
            value_armijo = value_tmp - value;
            if (value_armijo>=0)&&(counter>300)
                stop_flag = true;
                B = eye(31);
            end
            if (value_armijo > c*alpha) & (value_armijo < c*sigma*alpha)
                stop_flag = true;
            else
                alpha = alpha*beta;
            end
        end
        
        w = w + alpha.*d;
        prev_value = value;
        prev_grad = grad;
        [value, grad] = target_func(w, N, X, Y);
        value_array = [value_array; value];
        
        if (grad'*d) > sigma2*(prev_grad'*d)
            p = alpha.*d;
            q = grad - prev_grad;
            mu = p'*q;
            s = B*q;
            tau = s'*q;
            v = p/mu - s/tau;
            B = B + (p*p')/mu - (s*s')/tau + (v*v')*tau;
        end
    end
    min_x = w;
end
