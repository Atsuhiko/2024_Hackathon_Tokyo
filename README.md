# 2024 AI Hackathon Tokyo
This program was developed for 2024 AI Hackathon in Tokyo held on Sep.14-15.  
https://www.aifestival.jp/hackathon

Various AI models were used, such as  
- whisper by OpenAI  
  https://github.com/openai/whisper  
- Swallow Mistral by Tokyo Technology University  
  https://huggingface.co/tokyotech-llm/Swallow-MS-7b-v0.1  
- VOICEVOX  
  https://github.com/VOICEVOX
- SadTalker  
  https://github.com/OpenTalker/SadTalker

  To implement SadTalker in Gradio app, I modified the original inference.py code to inference6.py as attached.
  Based on Tokyo-tech llm's model, QLoRA instruction tunig was carried out using peft and TR libraries of Hugging Face.

  Please refer to original GitHub sites to prepare the implemantation environment.
