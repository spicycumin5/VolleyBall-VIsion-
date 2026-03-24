# VolleyBall-VIsion-

The hackathon project by Kyumin Han, Evan Inrig, and Jason Press (:

## Stuff Dr. Scalzo said in class

- Have a cracked presentation. Make it look *beautiful*
- One week is not much time. Have it look like it's promising
- We record a demo of what's possible, like a video. We can be a little selective of what we show off (:<
  - Hackathon doesn't need to deal with the longest video
  - Just a one minute clip is good enough
- Zaven et. al. are studying for a bunch of midterms over the week
  - They are using [[Vast.ai]] for their compute
  - They are also doing a preprocessed version
- A game-ified UI would be best

## Getting the Python working

Use a venv. I use `.venv` for my life, so I will assume you use `.venv` from now on. And also UNIX, because fuck Windows (use WSL).

``` sh
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

For running, make sure to `source .venv/bin/activate` to be in the virtual environment if you don't have the system packages globally.

Download `msmt_sbs_R101-ibn.pth` to `pthon/weights` to make `test.py` work.
