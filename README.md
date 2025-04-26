# ComfyUI Custom Dia

This is a ComfyUI integration of the [Dia TTS model](https://github.com/nari-labs/dia/).  
Many thanks to **nari-labs** for their fantastic work.

## Installation

Download the `.pth` and `.json` files from [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B/tree/main)  
Store them in any subfolder under `/models/` — the path is not hardcoded, and the node allows you to define it manually.  
(Default path: `/models/Dia/dia-v0_1.pth`)

## Modifications from the Original Repository

The original Dia API has been **slightly modified** to support **multi-channel audio inputs**.  
This allows for stereo files or tensors provided directly by ComfyUI nodes.

an extra node has been added to retime the output audio. See the example for usage (pitch preservation requires an extra package: librosa It's not in the requirements.txt to avoir intalling it for users who don't need that extra function)

## Usage

This is an **output node**, meaning it can be used standalone and queued without connections.  
In that case, you may want to enable `save_audio_file` to automatically save the result into ComfyUI’s output folder.

To use it in a pipeline, just connect the `audio` output to any compatible node.

### Speech Prompt

- Use the `text` field to define your dialogue, e.g.:
```
[S1] Hello.
[S2] Hi there! (laughs)
```

- Use `[S1]`, `[S2]`, etc. to switch speakers.
- Insert nonverbal tags (e.g. `(laughs)`, `(sighs)`) to enrich the audio.
- A list of available tags is provided in the third (inactive) text field.

![image](https://github.com/user-attachments/assets/d4a32dd7-0426-46c6-9685-2190dc7d6993)

## Voice Cloning

You can plug an `audio` tensor as input to enable **voice cloning**.  
In this case, it is strongly recommended to provide a **transcript** of the input audio in the `input_audio_transcript` field to improve results.

![image](https://github.com/user-attachments/assets/9bac4077-9a71-4ee1-a279-0773bb51a75a)


## Troubleshooting and side effects
As stated in the requirement.txt file, you will have to install 2 python packages: **descript-audio-codec** and **soundfile**

Under certain circonstances, **descript-audio-codec** installation could auomatically downgrade **protobuf** back into 3.19.6 which could make some other nodes crash on startup. If it ever happens, just upgrade protobuf by opening comfyUI terminal and run
```
pip install protobuf --upgrade
```

