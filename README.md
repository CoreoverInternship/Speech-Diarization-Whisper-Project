**Setting up the environment:**
You need to download CONDA(https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and CUDA(https://developer.nvidia.com/cuda-downloads) to get this working. 

Make sure both downloads are in the 'path' environment variable.

Create an env: 'conda create --name <env_name> python=3.10'. 

Activate the env: 'conda activate <env_name>'.

Download important packages using 'conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia',

Download other required packages from requirements.txt using 'pip install -r requiremnts.txt'

**Running the code:**
Run 'downloadModel.py' to download the model to be used in the diarization process.

Now we can run 'diarisation.py'. This file has a hardcoded directory from where it gets the audio files to diarize. You can also uncomment the code below the loop to diarise a file by itself instead.

**How the code works:**
Once the function is called to diarise a file, the name of the file passed in is cleaned which is used to open a txt file(for writing the transcript). 

Next, WhisperX loads the audio(pre-processing), prepares the diarization process by passing in the OpenAI API key, and calls WhisperX's diarization function on the processed audio. The function's return value is a type of table that has letters assorted(alphabetically) to the different speakers in the audio file, time segments for when they start and stop talking, and more useful information. The only information useful for us is the different speakers, and when start and stop talking.

To sort the audio and its corresponding(speech-to-text) text into a readable format we begin with entering a loop in which we get the start and end time of each spoken segment, and convert the time to seconds. NOTE: audio segments smaller or equal to 0.5 seconds are disregarded due to inconsistencies when converting that audio to text. Then we get the .wav version of the audio and call the 'speech_to_text' function to convert the audio to text. The function uses our trained model to return the text of the audio. Now that we have the text, in the same loop as before, we write the speaker and their corresponding text into a transcribe.txt file for each audio file. 

