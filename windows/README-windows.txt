LittleRun for Windows
=====================

Run LittleRun.exe. It needs Windows 10 or 11 (64-bit) and an internet connection the first time.

First run (setup)
  1. Choose CPU (recommended) or an NVIDIA GPU (experimental; needs a 16 GB card and driver 580+).
     The GPU reads prompts about 10x faster but writes at about CPU speed for now.
  2. Download the model (about 2.6 GB) or point LittleRun at a folder that already has it.
  3. Accept the model license and press Install.

Setup installs a private Python, PyTorch and the LittleRun backend into
%LOCALAPPDATA%\LittleRun. Nothing is installed system-wide and no admin rights are needed.

Using it
  - Chat: talk to the model. Complete: write the start of a text and LittleRun continues it.
  - Story chat: the pinned model was not trained for chat, so LittleRun writes the conversation as a short
    story in which a character answers. Pick Ada (AI assistant), Maya (park ranger), Leo (retired teacher)
    or write your own character in Settings > Character. Replies usually make sense; facts are often wrong.
    The first reply after picking a character takes about half a minute on CPU while the character loads.
  - Closing the window keeps the model loaded in the tray. Right-click the tray icon to quit.
  - Settings > Local API shows an OpenAI-style endpoint and key for other apps on this PC.

Model note
  The pinned checkpoint (LittleBit Qwen3 8B, 0.55 bits per weight) is a community release licensed
  CC BY-NC 4.0. It was not trained on chat data: with its own chat format the replies are gibberish,
  which is why LittleRun uses story chat for it. Complete mode works well.

Files
  %LOCALAPPDATA%\LittleRun\runtime   Python, PyTorch, backend (safe to delete; setup reinstalls it)
  %LOCALAPPDATA%\LittleRun\models    downloaded models
  %LOCALAPPDATA%\LittleRun\logs      setup.log, backend.log
  %LOCALAPPDATA%\LittleRun\chats.json

License
  LittleRun: PolyForm Noncommercial License 1.0.0 (LICENSE). See NOTICE for attribution.
