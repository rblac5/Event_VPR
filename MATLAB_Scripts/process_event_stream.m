clear
load('event_stream_mat.mat')

close all
event_stream = double(event_stream(1:1e6,:));
time_counts = (event_stream(:,1)-event_stream(1,1)); %us counts (/1e6 to get to seconds)

delta_time = diff(time_counts);

x_pix = event_stream(:,2);
y_pix = event_stream(:,3);
pol = event_stream(:,4); pol(pol==0) = -1;

pos_event_ind = find(pol == 1);
neg_event_ind = find(pol == -1);

x_pix_pos = x_pix(pos_event_ind);
y_pix_pos = y_pix(pos_event_ind);
time_pos = time_counts(pos_event_ind);

x_pix_neg = x_pix(neg_event_ind);
y_pix_neg = y_pix(neg_event_ind);
time_neg = time_counts(neg_event_ind);


% subplot(2,1,1); plot(time, '-*')
% subplot(2,1,2); plot(delta_time, '-*')
% duration = time(end)-time(1)

% event_matrix = zeros(max(y_pix)+1, max(x_pix)+1, time(end)+1);
% size(event_matrix)

figure(1)
plot3(x_pix_pos, time_pos, y_pix_pos, 'b.');
xlabel('x pixels'); ylabel('time'); zlabel('y pixels')
hold on
plot3(x_pix_neg, time_neg, y_pix_neg, 'r.')
%%
num_events_matrix = zeros(max(y_pix)+1, max(x_pix)+1);
for i = 0:max(x_pix)
    for j = 0:max(y_pix)
        num_events_at_coord = sum(x_pix==i & y_pix==j);
        num_events_matrix(j+1,i+1) = num_events_at_coord;
    end
end


%%
max_events_count = max(max(num_events_matrix));
[y_mat_ind,x_mat_ind] = find(num_events_matrix == max_events_count);
y_mat_ind = 14; x_mat_ind = 55;

max_y_pix = y_mat_ind(1)-1;
max_x_pix = x_mat_ind(1)-1;
% sum(x_pix == max_x_pix & y_pix == max_y_pix) %sanity check

max_pix_ind = find(x_pix==max_x_pix & y_pix==max_y_pix);
time_max_count = time_counts(max_pix_ind)/1e6;
pol_max_count = pol(max_pix_ind);

%%
figure(2);clf;
surf(num_events_matrix); xlabel('x pixels'); ylabel('y pixels'); zlabel('num events')
shading interp
hold on
plot3(x_mat_ind,y_mat_ind,num_events_matrix(y_mat_ind, x_mat_ind), 'r.', 'MarkerSize', 20)


%%
C = 1; %arbitrary number for C aka change in log intensity
grad_matrix = zeros(1,length(time_max_count)-1);
for i = 2:length(time_max_count)
    dt = time_max_count(i) - time_max_count(i-1);
    dy = C*pol_max_count(i);
    grad_matrix(i-1) = dy/dt;
end
reconstructed_log_intensity = cumsum(grad_matrix.*diff(time_max_count)');

%%
figure
subplot(3,1,1)
sgtitle(['(x,y) = (', num2str(max_x_pix), ',' num2str(max_y_pix), ')'])
plot(time_max_count, pol_max_count, 'k*')
title('Event polarity')
grid on

subplot(3,1,2)
plot(time_max_count(2:end), grad_matrix)
title('Log intensity gradient')
subplot(3,1,3)
plot(time_max_count(2:end), reconstructed_log_intensity, '-*')
title('Reconstructed Log intensity'); xlabel('Time (s)')


