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

Download `msmt_sbs_R101-ibn.pth` to `python/weights` to make `test.py` work.

### Downloading models

You can find models in the [Releases tab](https://github.com/spicycumin5/VolleyBall-VIsion-/releases). `yolo26x.pt` is a standard model and will be downloaded on first run if it is not found, and the same is true for the generic ID models. The custom ball and aciton tracker are required to be downloaded. Simply download them to the project directory.

If you want to use SAM3 model, download the model parts, and then run this in your project directory:

``` sh
cat sam3.pt.part-* > sam3.pt
```

And if you're not on UNIX, what are you doing with your life?

These models can be passed into `python/test.py` as flag paths. See the [Python README](./python/README.md) for more info.


## For the website

To run the site, navigate to the **frontend** file:
` cd frontend `
Currently we are running the project with Vite given the scope of the project. 
Install the dependencies using `npm i`. 
Afterwards, you can locally host the project using `npm run dev` and navigating to the localhost your terminal displays. 

There are **curently** 3 pages the site uses: a landing page, a homepage to browse videos and upload, and a specific video page. 
Pages can be accessed either in-site via navigation or via url like so (assuming 3000 as a baseline):
Landing page: `http://localhost:3000`
Homepage: `http://localhost:3000/home`
Video page: `http://localhost:3000/video`

