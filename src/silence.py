import numpy as np

def chunk_rms(step, duration, audio):
    duration_before = int(duration/2)
    duration_after = duration - duration_before
    starts = np.arange(duration_before, len(audio), step)
    rms = list(map(lambda start:(start, audio[start-duration_before:start+duration_after].rms), starts))
    return rms

def min_rms(rms, min_duration, max_duration):
    chunks = []
    i = 0
    last_silence = 0
    while(i<len(rms)):
        if ((rms[i][0] - last_silence) >= min_duration):
            best = rms[i]
            while((i+1)<len(rms) and (rms[i+1][0] - last_silence) <=max_duration):
                i+=1
                if (rms[i][1] < best[1]):
                    best = rms[i]
            chunks.append(best)
            last_silence = best[0]
        else:
            i+=1
    return chunks

def split_silence(audio, min_duration, max_duration, rms_step=5, rms_duration=100):
    rms = chunk_rms(rms_step, rms_duration, audio)
    print("Calculated rms for %d chunks " % (len(rms)))
    chunks = min_rms(rms, min_duration, max_duration)
    print("Extracted %d chunks based on silence" % (len(chunks)))
    return chunks