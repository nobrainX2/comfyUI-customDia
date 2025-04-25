import sys
import os
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

from .dia.model import Dia
import folder_paths

#avoid unnecessary errors being raised by comfyUI interface
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class DiaText2Speech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "models/Dia/dia-v0_1.pth"}),
                "save_audio_file": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "audio/dia"}),
                "speech": (
                    "STRING",
                    {
                        "multiline": True,
                        "label": "text",
                        "default": (
                            "[S1] Dia is an open weights text to dialogue model. \n"
                            "[S2] You get full control over scripts and voices. \n"
                            "[S1] Wow. Amazing. (laughs) \n"
                            "[S2] Try it now on Git hub or Hugging Face."
                        ),
                    },
                ),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 10.0}),
                "use_cfg_filter": ("BOOLEAN", {"default": True}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "cfg_filter_top_k": ("INT", {"default": 35, "min": 0, "max": 100}),
            },
            "optional": {
                "input_audio": ("AUDIO",),
                "input_audio_transcript": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "label": "input audio transcript",
                    },
                ),
                "available_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "(laughs), (clears throat), (sighs), (gasps), (coughs),\n"
                            "(singing), (sings), (mumbles), (beep), (groans),\n"
                            "(sniffs), (claps), (screams), (inhales), (exhales),\n"
                            "(applause), (burps), (humming), (sneezes), (chuckle), (whistles)"
                        ),
                        "label": "available tags",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/dia"
    OUTPUT_NODE = True

    def generate(
        self,
        model_path,
        save_audio_file,
        filename_prefix,
        speech,
        cfg_scale,
        temperature,
        top_p,
        use_cfg_filter,
        use_torch_compile,
        cfg_filter_top_k,
        input_audio=None,
        input_audio_transcript="",
        available_tags="",
    ):
        config_path  = Path(model_path).with_name("config.json")
        model = Dia.from_local(config_path=config_path , checkpoint_path=model_path)

        if input_audio is not None:
            waveform = input_audio["waveform"]  # [1, 2, N]
            sample_rate = input_audio["sample_rate"]

            inputsoundastensor = waveform
            if sample_rate != 44100:
                inputsoundastensor = torchaudio.functional.resample(inputsoundastensor, sample_rate, 44100)
            if inputsoundastensor.shape[1] == 2:
                inputsoundastensor = torch.mean(inputsoundastensor, dim=1, keepdim=True)

            inputsoundastensor = inputsoundastensor.to(model.device)
            audio_data = model.dac_model.preprocess(inputsoundastensor, 44100)
            _, encoded_frame, _, _, _ = model.dac_model.encode(audio_data)

            inputsoundastensor = encoded_frame.squeeze(0).transpose(0, 1)

            generated_audio = model.generate(
                input_audio_transcript + speech,
                None,
                cfg_scale,
                temperature,
                top_p,
                use_torch_compile,
                cfg_filter_top_k,
                audio_prompt=inputsoundastensor,
                audio_prompt_path=None,
                use_cfg_filter=use_cfg_filter,
                verbose=True,
            )
        else:
            generated_audio = model.generate(
                speech,
                None,
                cfg_scale,
                temperature,
                top_p,
                use_torch_compile,
                cfg_filter_top_k,
                audio_prompt=None,
                audio_prompt_path=None,
                use_cfg_filter=use_cfg_filter,
                verbose=True,
            )

        if save_audio_file:
            (
                full_output_folder,
                filename,
                counter,
                subfolder,
                filename_prefix,
            ) = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
            file = f"{filename}_{counter:05}_.mp3"
            fullfilepath = os.path.join(full_output_folder, file)
            sf.write(fullfilepath, generated_audio, 44100)

        output_tensor = torch.from_numpy(generated_audio).unsqueeze(0).unsqueeze(0)
        return ({"waveform": output_tensor, "sample_rate": 44100},)


NODE_CLASS_MAPPINGS = {
    "Dia text to speech": DiaText2Speech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dia text to speech": "Dia text to speech",
}
