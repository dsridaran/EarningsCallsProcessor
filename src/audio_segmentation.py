import numpy as np
import os
from pydub import AudioSegment

def split_mp3(filename, num_segments):
    ''' Evenly split an mp3 file into num_segments segments '''
    os.makedirs(filename[:-4], exist_ok=True)
    audio = AudioSegment.from_mp3(filename)
    duration_ms = len(audio)
    segment_length = duration_ms // num_segments
    for i in range(0, duration_ms, segment_length):
        segment = audio[i:i + segment_length]
        segment_filename = filename[:-4] + f"/segment_{i // segment_length + 1}.mp3"
        segment.export(segment_filename, format="mp3")
        print(f"Exported {segment_filename}")

# go through all mp3 files in the directory and subdirectories and split them
def split_all(path, num_segments=15):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp3'):
                filename = file.split('/')[-1][:-4]
                company = root.split('/')[-1]
                output_dir = '../Test/' + company + '/' + filename
                os.makedirs(output_dir, exist_ok=True)
                audio = AudioSegment.from_mp3(root + '/' + file)
                duration_ms = len(audio)
                segment_length = duration_ms // num_segments
                for i in range(0, duration_ms, segment_length):
                    segment = audio[i:i + segment_length]
                    segment_filename = output_dir + f"/segment_{i // segment_length + 1}.mp3"
                    segment.export(segment_filename, format="mp3")
                print(f"Exported {output_dir}")
                # delete segment 16 if exists
                if os.path.exists(output_dir + '/segment_16.mp3'):
                    os.remove(output_dir + '/segment_16.mp3', )


if __name__ == '__main__':
    split_all('../Audio/', 15)
