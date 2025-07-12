from transformers import MarianMTModel, MarianTokenizer

# 假设你有 Whisper 的分段输出如下：
segments = [
    {"start": 0.0, "end": 2.3, "text": "こんにちは"},
    {"start": 2.4, "end": 4.7, "text": "お元気ですか？"}
]

# 翻译模型
model_name = 'Helsinki-NLP/opus-mt-ja-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


# 生成字幕行
def srt_time_format(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


srt_lines = []
for i, seg in enumerate(segments, 1):
    inputs = tokenizer(seg["text"], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    zh_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    start = srt_time_format(seg["start"])
    end = srt_time_format(seg["end"])
    srt_lines.append(f"{i}\n{start} --> {end}\n{zh_text}\n")

# 保存为 .srt 文件
with open("output.srt", "w", encoding="utf-8") as f:
    f.writelines(srt_lines)
