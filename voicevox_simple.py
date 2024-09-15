from pathlib import Path
import voicevox_core
from voicevox_core import AccelerationMode, VoicevoxCore
import gradio as gr
import whisper
import os
import io
import torch
import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# LLMモデルの定義
# モデルとトークナイザの準備
#   modelの指定
model_id = "tokyotech-llm/Swallow-MS-7b-instruct-v0.1"
peft_name = "../付喪神ジェネレータ/QLoRA_models/black_mistral"

# 量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# モデルの設定
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    # token=token, # HuggingFaceにログインしておけば不要
    quantization_config=bnb_config, # 量子化
    device_map='auto',
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
)
# tokenizerの設定
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="right",
    add_eos_token=True
)
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# ファインチューニングモデルの作成
model = PeftModel.from_pretrained(base_model, peft_name)

# 文章生成関数（EZO用）--> Llama3.1でもこのコードは有効だと分かった
def generate(prompt, max_tokens=32, temperature=1, top_p=0.9, top_k=30):

    prompt = f"""system prompt:あなたは缶コーヒーを飲むユーザーと対話するAIです。ユーザーの発言に対し大阪弁で応答し、その後、何かユーザーへ質問してください。
    user prompt: {prompt}"""
    
    messages = [
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones(input_ids.shape, dtype=torch.long).to(model.device),
    )
    response = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response, skip_special_tokens=True)
    return response_text


# 音声からテキストへの変換関数（whisperライブラリ版）
whisper_model = whisper.load_model("large") # 継之助なので large にする
# whisper_model = whisper.load_model("medium")
def speech_to_text(input_audio):
    response = whisper_model.transcribe(input_audio, verbose=True, language='ja')
    return response["text"]

# VOICEVOXによる音声合成
def VOICEVOX(text, SPEAKER_ID=13):
    open_jtalk_dict_dir = './open_jtalk_dic_utf_8-1.11'
    acceleration_mode = AccelerationMode.AUTO
    core = VoicevoxCore(
        acceleration_mode=acceleration_mode, open_jtalk_dict_dir=open_jtalk_dict_dir
    )
    core.load_model(SPEAKER_ID)
    audio_query = core.audio_query(text, SPEAKER_ID)
    wav = core.synthesis(audio_query, SPEAKER_ID)
    
    # 音声データを一時ファイルに保存
    output_file = "temp.wav"
    with open(output_file, "wb") as f:
        f.write(wav)
    
    return output_file

# 応答を生成し、フィルタリング処理を行う関数
def process_text_response(input_text):
    # テキスト処理とフィルタリング
    code_regex = re.compile(r'[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠！｀ 　＋￥％abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]')
    response_text = code_regex.sub('', generate(input_text)).replace("\n","")
    print("original response:", response_text) 

    # 文法修正やランダムな応答を挿入
    response_text = response_text.replace("ですが","やけど").replace("ありますが","あるんやけど").replace("あり。","ありますねん。").\
                                  replace("しているのやねん。","してるんねん。").replace("わい、思いますねん。","").replace("わい", "俺")

    candidates = ["で、どう思う？", "そうなんや、ええな", "ほんま？", "まじで？", "オチは？"]
    sample = random.choice(candidates)
    response_text = response_text.replace("知らんけど。", sample)
    
    print("===============================")
    print("processed response text: " + response_text)
    return response_text

# Gradio用のアプリケーションインターフェース
def process_audio(input_audio, speaker_id):
    # マイクからの音声をテキストに変換
    text = speech_to_text(input_audio)
    
    # テキストの処理と修正
    processed_text = process_text_response(text)
    
    # Voicevoxで音声を合成
    output_audio = VOICEVOX(processed_text, speaker_id)
    
    return output_audio

# Gradio インターフェースの設定
iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone"], label="Input audio", type="filepath"),
        gr.Dropdown([11, 12, 13], value=13, label="話者（例: 青山龍星）"),  # 話者選択
    ],
    outputs=gr.Audio(type="filepath", label="生成された音声", autoplay=True),
    title="WhisperとVoicevoxを使った音声変換アプリ",
    description="マイクで音声を入力し、Whisperでテキストに変換後、Voicevoxで音声合成します。"
)

# アプリの起動
iface.launch()
