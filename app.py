import streamlit as st
import os, json
from faster_whisper import WhisperModel

import time, tempfile
import ffmpeg
from moviepy.editor import TextClip, CompositeVideoClip, ColorClip, VideoFileClip, ColorClip

import numpy as np



start = time.time()

directory = tempfile.gettempdir()

def create_audio(videofilename):
    audiofilename = videofilename.replace(".mp4", '.mp3')

    # Create the ffmpeg input stream
    input_stream = ffmpeg.input(videofilename)

    # Extract the audio stream from the input stream
    audio = input_stream.audio

    # Save the audio stream as an MP3 file
    output_stream = ffmpeg.output(audio, audiofilename)

    # Overwrite output file if it already exists
    output_stream = ffmpeg.overwrite_output(output_stream)

    ffmpeg.run(output_stream)

    return audiofilename



def transcribe_audio(whisper_model, audiofilename):
    segments, info = whisper_model.transcribe(audiofilename, word_timestamps=True)


    segments = list(segments)  # The transcription will actually run here.

    wordlevel_info = []

    for segment in segments:

        for word in segment.words:
            wordlevel_info.append({'word': word.word.upper(), 'start': word.start, 'end': word.end})

    return wordlevel_info


def split_text_into_lines(data, v_type , MaxChars):
    MaxDuration = 2.5

    # Split if nothing is spoken (gap) for these many seconds
    MaxGap = 1.5

    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0

    for idx, word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        line.append(word_data)
        line_duration += end - start

        temp = " ".join(item["word"] for item in line)

        # Check if adding a new word exceeds the maximum character count or duration
        new_line_chars = len(temp)

        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars
        if idx > 0:
            gap = word_data['start'] - data[idx - 1]['end']
            # print (word,start,end,gap)
            maxgap_exceeded = gap > MaxGap
        else:
            maxgap_exceeded = False

        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line
                }
                subtitles.append(subtitle_line)
                line = []
                line_duration = 0
                line_chars = 0

    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line
        }
        subtitles.append(subtitle_line)

    return subtitles


def create_caption(textJSON, framesize, v_type,highlight_color, fontsize, color,font="Roboto/Roboto-Bold.ttf",
                   stroke_color='black', stroke_width=2.6):
    wordcount = len(textJSON['textcontents'])
    full_duration = textJSON['end'] - textJSON['start']

    word_clips = []
    xy_textclips_positions = []

    x_pos = 0
    y_pos = 0
    line_width = 0  # Total width of words in the current line
    frame_width = framesize[0]
    frame_height = framesize[1]

    x_buffer = frame_width * 1 / 10

    max_line_width = frame_width - 2 * (x_buffer)
    # st.write(frame_height)
    # if v_type.lower() != 'reels':
    #     fontsize = int(frame_height * 0.065)  # 7.5 percent of video height
    # else:
    #     fontsize = int(frame_height * 0.04)  # 5 percent of video height for Reels

    fontsize = int(frame_height * fontsize/100)  # 5 percent of video height for Reels
    # st.write(fontsize)
    space_width = 0
    space_height = 0

    for index, wordJSON in enumerate(textJSON['textcontents']):
        bold_font_path = "Poppins/Poppins-ExtraBold.ttf"

        duration = wordJSON['end'] - wordJSON['start']
        word_clip = TextClip(wordJSON['word'], font=bold_font_path, fontsize=fontsize, color=color,
                             stroke_color=stroke_color,
                             stroke_width=stroke_width , kerning=-5).set_start(textJSON['start']).set_duration(
            full_duration)

        word_clip_space = TextClip(" ", font=bold_font_path, fontsize=fontsize, color=color ,kerning=-5).set_start(
            textJSON['start']).set_duration(full_duration)
        word_width, word_height = word_clip.size
        space_width, space_height = word_clip_space.size
        space_width = 0
        space_height = 0
        if line_width + word_width + space_width <= max_line_width:
            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos": x_pos,
                "y_pos": y_pos,
                "width": word_width,
                "height": word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos, y_pos))
            word_clip_space = word_clip_space.set_position((x_pos + word_width, y_pos))

            x_pos = x_pos + word_width + space_width
            line_width = line_width + word_width + space_width
        else:
            # Move to the next line
            x_pos = 0
            y_pos = y_pos + word_height + 10
            line_width = word_width + space_width

            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos": x_pos,
                "y_pos": y_pos,
                "width": word_width,
                "height": word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos, y_pos))
            word_clip_space = word_clip_space.set_position((x_pos + word_width, y_pos))
            x_pos = word_width + space_width

        word_clips.append(word_clip)
        word_clips.append(word_clip_space)

    for highlight_word in xy_textclips_positions:
        word_clip_highlight = TextClip(highlight_word['word'], font=bold_font_path, fontsize=fontsize, color=highlight_color,
                                       stroke_color=stroke_color, stroke_width=stroke_width , kerning=-5).set_start(
            highlight_word['start']).set_duration(highlight_word['duration'])
        word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
        word_clips.append(word_clip_highlight)

    return word_clips, xy_textclips_positions


def get_final_cliped_video(videofilename, linelevel_subtitles, v_type , subs_position , highlight_color, fontsize, opacity ,color):
    input_video = VideoFileClip(videofilename)
    frame_size = input_video.size

    all_linelevel_splits = []

    for line in linelevel_subtitles:
        out_clips, positions = create_caption(line, frame_size, v_type , highlight_color , fontsize, color)
#to increase space horizontally
        max_width = 0
        max_height = 0

        for position in positions:
            # print (out_clip.pos)
            # break
            x_pos, y_pos = position['x_pos'], position['y_pos']
            width, height = position['width'], position['height']

            max_width = max(max_width, x_pos + width)
            max_height = max(max_height, y_pos + height)

        color_clip = ColorClip(size=(int(max_width * 1.1), int(max_height * 1.1)),
                               color=(64, 64, 64))
        color_clip = color_clip.set_opacity(opacity)
        color_clip = color_clip.set_start(line['start']).set_duration(line['end'] - line['start'])

        clip_to_overlay = CompositeVideoClip([color_clip] + out_clips)

        if subs_position == "bottom75":
            clip_to_overlay = clip_to_overlay.set_position(('center', 0.75), relative=True)
        else:
            clip_to_overlay = clip_to_overlay.set_position(subs_position)

        all_linelevel_splits.append(clip_to_overlay)

    input_video_duration = input_video.duration

    final_video = CompositeVideoClip([input_video] + all_linelevel_splits)

    # Set the audio of the final video to be the same as the input video
    final_video = final_video.set_audio(input_video.audio)
    destination = os.path.join(directory, 'output.mp4')
    # Save the final clip as a video file with the audio included
    final_video.write_videofile(destination, fps=24, codec="libx264", audio_codec="aac")
    # st.video(final_video)
    return destination


def add_subtitle(videofilename, audiofilename, v_type, subs_position, highlight_color, fontsize, opacity, MaxChars , color, wordlevel_info):
    print("video type is: " + v_type)
    print("Video and Audio filesare : ", videofilename, audiofilename)


    print("word_level: ", wordlevel_info)

    linelevel_subtitles = split_text_into_lines(wordlevel_info, v_type, MaxChars)
    print("line_level_subtitles :", linelevel_subtitles)
    st.session_state.linelevel_subtitles = linelevel_subtitles

    for line in linelevel_subtitles:
        json_str = json.dumps(line, indent=4)
        print("whole json: ", json_str)
    outputfile = get_final_cliped_video(videofilename, linelevel_subtitles, v_type, subs_position,highlight_color , fontsize, opacity , color)
    return outputfile


def show_audio_video(audiofilename, videofilename):
    st.video(videofilename)





@st.cache_resource
def load_model():
    print('Loading the Whisper Model into the App............')

    model_size = "base"
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    except RuntimeError as e:
        model = WhisperModel(model_size)
        print("Error: ", e)

    st.success("Loaded model!")
    return model




st.title('Autocaption by [Fictions.ai](https://fictions.ai)')
st.write("[GitHub Repo](https://github.com/fictions-ai/autocaption) - [x.com/thibaudz](https://x.com/thibaudz)")

st.title('\n')

if 'whisp_model' not in st.session_state:
    print("Downloading dependecies...")

    try:
        # install_img_magic_commands_linux()
        print("Dependencise for video editing downloaded successfully.")
        st.session_state.img_magik = True
    except Exception as e:
        print('Some Errors ocurred while loading: ', e)

    model = load_model()
    st.session_state.whisp_model = model
    print('whisper Loaded.')

video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

col1, col2 = st.columns(2)

subs_position = st.selectbox(
   "Select the subtitles position",
   ( "bottom75", "center" , "top", "bottom", "left", "right" ),
   placeholder="Select subtitless position...",
)

color = st.text_input("Select the caption color", "white")
v_type = st.radio("Select the Video ratio", ['9x16', "other aspect ratio"])

# st.write(v_type)
# if st.button('Advance Options'):
with st.expander("More Options"):

    highlight_color = st.text_input("Highlight Color", "yellow")
    if v_type != "9x16":
        fontsize = st.number_input("Fontsize (7.0 is ideal for videos)", min_value=1.0 , value=7.0)
        MaxChars = st.number_input("Max Characters space for subtitles  (20 is ideal for videos, increasing this will increase the subtitles width)", min_value=1, value=20)

    else:
        fontsize = st.number_input("Fontsize (4.0 is ideal for reels)", min_value=1.0 , value=4.0)
        MaxChars = st.number_input("Max Characters space for subtitles  (10 is ideal for reels, increasing this will increase the subtitles width)", min_value=1, value=10)
    opacity = st.number_input("Opacity for the subtitles background (Set it to 0 for no background) Range: 0 - 1.0 ", min_value=0.0, value=0.0)
button = st.button("Generate Video")
try:
    if video_file is not None and button:
        video_url = None
        clip_sbtitle = False

        videofilename = os.path.join(directory, video_file.name)
        with open(videofilename, "wb") as f:
            f.write(video_file.read())

        st.session_state.videofile_path = videofilename
        st.session_state.audio = None

    if clip_sbtitle:
        video_file = None
        # download_video(video_url)
        videofilename = os.path.join(directory, "video.mp4")
        st.session_state.videofile_path = videofilename
        st.session_state.audio = None

        if 'outputfile_path' in st.session_state and st.session_state.outputfile_path is not None:
            st.session_state.outputfile_path = None

    if "edit" not in st.session_state:
        st.session_state.edit = False


    if video_file and subs_position:

        # if 'outputfile_path'  in st.session_state and st.session_state.outputfile_path is not None:
        #   st.session_state.outputfile_path = None

        videofilename = st.session_state.videofile_path
        # st.write("Title :", videofilename.split('\\')[-1])

        st.video(videofilename)

        cols = st.columns(2)



        # with cols[1]:

        st.session_state.edit = True

        try:
            if st.session_state.audio is None:
                with st.spinner('Converting Audio.......'):
                    audiofilename = create_audio(videofilename)
                    st.success('Audio Created')
                    st.session_state.audio = audiofilename

            audiofilename = st.session_state.audio
            with st.spinner('Editing Video.......'):
                if 'img_magik' in st.session_state:
                    try:
                        model = st.session_state.whisp_model
                        wordlevel_info = transcribe_audio(model, audiofilename)
                        print(type(wordlevel_info))
                        import ast
                        wordlevel_info = st.text_area("Edit the transcription (optional):", wordlevel_info)
                        wordlevel_info = ast.literal_eval(wordlevel_info)
                        print(type(wordlevel_info))

                        outputfile = add_subtitle(videofilename, audiofilename, v_type, subs_position, highlight_color, fontsize, opacity, MaxChars , color , wordlevel_info)

                    except Exception as e:
                        outputfile = videofilename
                        print('There is an erro:', e)
                        st.session_state.add_subtitles = False

                    st.session_state.add_subtitles = True
                else:
                    outputfile = videofilename
                st.session_state.outputfile_path = outputfile
                st.success('Video Created')
        except Exception as e:
            st.write('There is an error, please refresh the page or upload other video:', e)

        if 'outputfile_path' in st.session_state and st.session_state.outputfile_path is not None:
            show_audio_video(st.session_state.audio, st.session_state.outputfile_path)

except Exception as e:
    print(e)
