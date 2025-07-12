import whisper
import subprocess
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer


def extract_audio(video_path, wav_path):
    # subprocess.run([
    #     "ffmpeg", "-y", "-i", video_path,
    #     "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    #     wav_path
    # ], check=True)
    # docker run --rm -v ${PWD}:/tmp jrottenberg/ffmpeg -i /tmp/30757228173-1-16.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/audio.wav
    subprocess.run([
        "docker", "run", "--rm", "-v", f"{video_path}:/tmp",
        "jrottenberg/ffmpeg", "-i", "/tmp/30757228173-1-16.mp4", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wav_path])


def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def translate(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def generate_srt(segments, translations, output_path="output.srt"):
    lines = []
    for i, (seg, trans) in enumerate(zip(segments, translations), 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        lines.append(f"{i}\n{start} --> {end}\n{trans}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def whisper_transcribe(wav_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(wav_path)
    return result

def main(video_path):
    wav_path = r"D:/data/code/PyTorch2_Study/video2text/audio.wav"
    # print("🔊 提取音频...")
    # extract_audio(video_path, wav_path)

    print("🧠 Whisper 识别中...")
    # model = whisper.load_model("base")  # 可选: tiny, base, small, medium, large
    # result = model.transcribe(wav_path, language="ja")
    result = whisper_transcribe(wav_path)

    segments = result['segments']
    ja_texts = [seg['text'] for seg in segments]

    print("🌐 翻译中（日→中）...")
    trans_model_name = "Helsinki-NLP/opus-mt-ja-zh"
    tokenizer = MarianTokenizer.from_pretrained(trans_model_name)
    translator = MarianMTModel.from_pretrained(trans_model_name)

    zh_texts = translate(ja_texts, translator, tokenizer)

    print("📄 正在生成 SRT 字幕...")
    generate_srt(segments, zh_texts)

    print("✅ 字幕已生成：output.srt")


if __name__ == "__main__":
    import sys

    main("")
    # if len(sys.argv) < 2:
    #     print("用法：python generate_subtitles.py video.mp4")
    # else:
    #     main(sys.argv[1])
