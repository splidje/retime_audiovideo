#!/usr/bin/env python

class Curve:
    def __init__(self, data):
        data = sorted(data)
        self._curve_sections = tuple(
            CurveSection(data[i], data[i+1]) for i in range(len(data) - 1)
        )

    def value_at(self, x):
        for curve_section in self._curve_sections:
            if x >= curve_section.min_x and x <= curve_section.max_x:
                return curve_section.value_at(x)

class CurveSection:
    def __init__(self, p1, p2):
        self.min_x = p1[0]
        self.max_x = p2[0]
        if len(p1) == 3:
            #a * p1[0]^3 + b * p1[0]^2 + c * p1[0] + d = p1[1]
            #a * p2[0]^3 + b * p2[0]^2 + c * p2[0] + d = p2[1]
            #3 * a * p1[0]^2 + 2 * b * p1[0] + c = p1[2]
            #3 * a * p2[0]^2 + 2 * b * p2[0] + c = p2[2]
            self._a = (p1[0]*(p1[2] + p2[2]) - (p1[2] + p2[2])*p2[0] - 2*p1[1] + 2*p2[1])/(p1[0]**3 - 3*p1[0]**2*p2[0] + 3*p1[0]*p2[0]**2 - p2[0]**3)
            #self._b = -(2*self._a*p1[0]**3 - 3*self._a*p1[0]**2*p2[0] + self._a*p2[0]**3 - p1[0]*p1[2] + p1[2]*p2[0] + p1[1] - p2[1])/(p1[0]**2 - 2*p1[0]*p2[0] + p2[0]**2)
            #self._c = -(self._a*p1[0]**3 - self._a*p2[0]**3 + self._b*p1[0]**2 - self._b*p2[0]**2 - p1[1] + p2[1])/(p1[0] - p2[0])
            #self._d = -self._a*p1[0]**3 - self._b*p1[0]**2 - self._c*p1[0] + p1[1]
        elif len(p1) == 2:
            #b * p1[0]^2 + c * p1[0] + d = p1[1]
            #b * p2[0]^2 + c * p2[0] + d = p2[1]
            #2 * b * p2[0] + c = p2[2]
            self._a = 0
        self._b = -(self._a*p1[0]**3 + 2*self._a*p2[0]**3 - (3*self._a*p2[0]**2 - p2[2])*p1[0] - p2[0]*p2[2] - p1[1] + p2[1])/(p1[0]**2 - 2*p1[0]*p2[0] + p2[0]**2)
        self._c = -(self._a*p1[0]**3 - self._a*p2[0]**3 + self._b*p1[0]**2 - self._b*p2[0]**2 - p1[1] + p2[1])/(p1[0] - p2[0])
        self._d = -self._a*p1[0]**3 - self._b*p1[0]**2 - self._c*p1[0] + p1[1]

    def value_at(self, x):
        return self._a * x**3 + self._b * x**2 + self._c * x + self._d

def speed_audio(data, speed):
    data_sped = numpy.empty((int(round(data.shape[0] / speed)), data.shape[1]))
    for ch in range(data.shape[1]):
        ch_sped = librosa.effects.time_stretch(data[:, ch], speed)
        data_sped[:int(round(data.shape[0] / speed)), ch] = ch_sped
    return data_sped

if __name__ == "__main__":
    import argparse
    import sys
    import re
    import collections

    parser = argparse.ArgumentParser(description='retime whole video and audio using given mapping')
    parser.add_argument('in_path')
    func_args = parser.add_mutually_exclusive_group(required=True)
    func_args.add_argument('-c', '--curve', nargs="+", help="comma separated pairs or triples of floats. <out time> <in time> [<gradient>]. (secs)")
    func_args.add_argument('-e', '--expr', nargs="+", help="python expression given t as out time and evaluating to in time. (secs)")
    parser.add_argument('-g', '--graph', action='store_true', help="Show a graph of the curve before processing.")
    parser.add_argument('out_path')
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path

    print("Importing modules...", file=sys.stderr)
    import moviepy.editor
    import numpy
    import librosa
    import cv2

    if args.curve is not None:
        map_func = Curve(tuple(
            tuple(map(float, re.split("\s+", p_str)))
            for p_str
            in re.split(
                "\s*,\s*",
                " ".join(args.curve).strip()
            )
        )).value_at
    else:
        expr = " ".join(args.expr)
        map_func = lambda t: eval(
            expr,
            dict(
                t=t,
                sin=numpy.sin,
                cos=numpy.cos,
                tan=numpy.tan,
            ),
        )

    print("Reading media properties...", file=sys.stderr)

    in_video = moviepy.editor.VideoFileClip(in_path)
    in_audio = in_video.audio
    samp_per_frame = in_audio.fps / in_video.fps

    print("Calculating domain...", file=sys.stderr)

    max_in_s = int(in_video.duration * in_audio.fps) - 1
    out_s = 0
    jump = max_in_s
    good_ses = {}
    max_out_s = 0
    all_good = True
    prev_good = True
    jump = max_in_s
    while True:
        in_t = map_func(out_s / in_audio.fps)
        if in_t is not None:
            in_s = int(round(in_t * in_audio.fps))
        if in_t is None or in_s > max_in_s:
            all_good = False
            if prev_good:
                jump = -jump
            prev_good = False            
        else:
            if not prev_good:
                jump = -jump
            if out_s > max_out_s:
                max_out_s = out_s
                poss_max_in_s = in_s
            prev_good = True
        if not all_good:
            jump //= 2
        if jump == 0:
            max_in_s = poss_max_in_s
            break
        out_s += jump
    print((max_in_s, max_out_s))

    if args.graph:
        from matplotlib import pyplot
        x = range(0, max_out_s+1)
        y = [map_func(xx / in_audio.fps) * in_audio.fps for xx in x]
        pyplot.plot(x,y)
        pyplot.show()

    print("Retiming audio...", file=sys.stderr)

    new_audio = numpy.zeros((max_out_s + 1, in_audio.nchannels))

    out_s = 0
    while round(out_s) <= max_out_s:
        in_s = int(round(map_func(out_s / in_audio.fps) * in_audio.fps))
        if in_s == max_in_s:
            break
        next_out_s = out_s + samp_per_frame
        if next_out_s > max_out_s:
            out_s_end = max_out_s + 1
            in_s_end = max_in_s + 1
        else:
            out_s_end = int(round(next_out_s))
            in_s_end = int(round(map_func(next_out_s / in_audio.fps) * in_audio.fps))
        out_s_rnd = int(round(out_s))
        is_neg_speed = in_s > in_s_end
        if is_neg_speed:
            in_s, in_s_end = in_s_end, in_s
        in_s_len = in_s_end - in_s
        out_s_len = out_s_end - out_s_rnd
        speed = in_s_len / out_s_len
        in_chunk = in_audio.subclip(
            in_s / in_audio.fps,
            min( (in_s_end-1) / in_audio.fps, in_audio.duration ),
        ).to_soundarray()
        if in_chunk.shape[0] < in_s_len:
            new_in_chunk = numpy.zeros((in_s_len, in_audio.nchannels))
            new_in_chunk[:in_chunk.shape[0], :] = in_chunk
            in_chunk = new_in_chunk
        out_chunk = speed_audio(in_chunk, speed)
        if is_neg_speed:
            out_chunk = out_chunk[::-1,:]
        new_audio[out_s_rnd:out_s_end, :] = out_chunk
        print("{} / {} {}\t\r".format(out_s, max_out_s, speed), end="", file=sys.stderr)
        out_s = next_out_s

    def audio_make_frame(t):
        if isinstance(t, collections.Iterable):
            t_list = t
            audio_frame = numpy.empty((len(t_list), in_audio.nchannels))
            for i, t in enumerate(t_list):
                audio_frame[i] = new_audio[int(round(t * in_audio.fps))]
            return audio_frame
        else:
            return new_audio[int(round(t * in_audio.fps))]

    out_audio = moviepy.editor.AudioClip(audio_make_frame, duration=(max_out_s + 1) / in_audio.fps, fps=in_audio.fps)

    print("Retiming video and writing out...", file=sys.stderr)

    in_video_cv2 = cv2.VideoCapture(in_path)
    in_video_cv2_pos = 0

    def video_make_frame(out_t):
        global in_video_cv2_pos
        in_f = int(round(map_func(out_t) * in_video.fps))
        next_out_t = out_t + 1 / in_video.fps
        if next_out_t * in_audio.fps > max_out_s:
            in_f_end = int(in_video.duration * in_video.fps) + 1
        else:
            in_f_end = int(round(map_func(next_out_t) * in_video.fps))
        if in_f > in_f_end:
            in_f, in_f_end = in_f_end, in_f
        num_f = in_f_end - in_f
        im = numpy.zeros((in_video.h, in_video.w, 3))
        if in_video_cv2_pos != in_f:
            in_video_cv2.set(cv2.CAP_PROP_POS_FRAMES, in_f)
        for f in range(in_f, in_f_end):
            res, im_f = in_video_cv2.read()
            if res:
                im_f = cv2.cvtColor(im_f, cv2.COLOR_BGR2RGB)
                im += im_f / num_f
        in_video_cv2_pos = in_f_end
        return numpy.clip(im, 0, 255).astype(numpy.uint8)
    

    out_video = moviepy.editor.VideoClip(make_frame=video_make_frame, duration=out_audio.duration)
    out_video = out_video.set_audio(out_audio)

    out_video.write_videofile(out_path, fps=in_video.fps)
