%%
clear

[x_mosaic, y_mosaic] = meshgrid(0:100);
z_mosaic = create_mosaic(x_mosaic, y_mosaic);
z_max = max(max(z_mosaic));

event_cam = init_event_cam(1, 1);


plot_state(event_cam, x_mosaic, y_mosaic, z_mosaic)

t_lim = 30; %time steps
delta_t = 1;
event_matrix = zeros(event_cam.size, event_cam.size, t_lim);
for t = 1:delta_t:t_lim
    event_cam = move_event_cam(delta_t, event_cam);
%     event_matrix(:,:,t) = ((event_cam.mosaic_current - event_cam.mosaic_prev)>=C) + ((event_cam.mosaic_current - event_cam.mosaic_prev)<=C*-1)*-1;
    plot_state(event_cam, x_mosaic, y_mosaic, z_mosaic)
    figure(2)
    surf(double(event_cam.event_mat)); zlim([-1 1])
    
    pause(0.5)
end

% function [x,y] = event_cam_trajectory(t)
%     x = t;
% end
%%
function event_cam = move_event_cam(delta_t, event_cam)
    
    updated_x0 = event_cam.x0 + delta_t*event_cam.dxdt;
    updated_y0 = event_cam.y0 + delta_t*event_cam.dydt;
    
    event_cam = update_event_cam(updated_x0,updated_y0, event_cam);
end

function mosaic = create_mosaic(x, y)
%     mosaic = 0.5*x-20;
    mosaic = 0.5*x.*(x < 20);
end

function event_cam = init_event_cam(x0, y0)
    event_cam.size = 10;
    event_cam.x0 = x0;
    event_cam.y0 = y0;
    event_cam.x1 = event_cam.x0+event_cam.size-1;
    event_cam.y1 = event_cam.y0+event_cam.size-1;
    [event_cam.x, event_cam.y] = meshgrid(event_cam.x0:event_cam.x1, event_cam.y0:event_cam.y1);

%     event_cam.mosaic_current = create_mosaic(event_cam.x, event_cam.y);
    event_cam.mosaic_prev = zeros(size(event_cam.x));
    event_cam.is_pos_event_mat = zeros(size(event_cam.x));
    event_cam.is_neg_event_mat = zeros(size(event_cam.x));
    event_cam.is_event_mat = zeros(size(event_cam.x));

    event_cam.event_mat = zeros(size(event_cam.x));
    event_cam.C = 1;
   
    event_cam.dxdt = 1;
    event_cam.dydt = 0;
end

function event_cam = update_event_cam(x0,y0, event_cam)
    event_cam.size = 10;
    event_cam.x0 = x0;
    event_cam.y0 = y0;
    event_cam.x1 = event_cam.x0+event_cam.size-1;
    event_cam.y1 = event_cam.y0+event_cam.size-1;
    [event_cam.x, event_cam.y] = meshgrid(event_cam.x0:event_cam.x1, event_cam.y0:event_cam.y1);
    
    mosaic_current_temp = create_mosaic(event_cam.x, event_cam.y);

    event_cam.is_pos_event_mat = (mosaic_current_temp - event_cam.mosaic_prev) >= event_cam.C;
    event_cam.is_neg_event_mat = (mosaic_current_temp - event_cam.mosaic_prev) <= event_cam.C*-1;
    event_cam.is_event_mat = event_cam.is_pos_event_mat + event_cam.is_neg_event_mat;
    event_cam.event_mat = zeros(size(event_cam.x)) + event_cam.is_pos_event_mat + -1*event_cam.is_neg_event_mat;

    event_cam.mosaic_prev = event_cam.mosaic_prev.*(~event_cam.is_event_mat) + mosaic_current_temp.*(event_cam.is_event_mat);
%     event_cam.mosaic_current = 
  
end



function plot_state(event_cam, x_mosaic, y_mosaic, z_mosaic)
    figure(1); clf;
    surf(x_mosaic, y_mosaic, z_mosaic); xlabel('x'); ylabel('y'); zlabel('Log intensity');
    shading interp
    hold on
    plot3([event_cam.x0, event_cam.x1, event_cam.x1, event_cam.x0, event_cam.x0], [event_cam.y0, event_cam.y0, event_cam.y1, event_cam.y1, event_cam.y0], max(max(z_mosaic))*(ones(1,5)), 'r-')
    view(2); axis equal; xlim([0, 100]); ylim([0, 100])
end