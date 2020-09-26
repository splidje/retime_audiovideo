# retime_audiovideo
Run retime_audiovideo.py -h to see usage.

You can give it points on a curve or a python expression describing how to map an out time to an in time (in seconds).

The curve option takes comma separated points on the curve and gradient. The first point can be without gradient:
e.g.
--curve 0 0, 10 110 1, 20 120 1
Which describes a curve making the first 110 seconds of the input video fit into the first 10 seconds of the output video, gradually slowing down to normal speed, then the last 10 seconds is at normal speed