clear 
close all
clc

load seis_data

sig = seis_data;

dt = 0.002;
tt = dt*[1:length(sig)];

figure
plot(tt,sig,'k');
xlabel 'time (s)'
ylabel 'amplitude'
box on

