# NOTE: must install 
#   R https://cran.r-project.org/bin/macosx/
#     lmer
#   $ R
#   > install.packages("lme4")


# to get tf on nlp machines:
#   pip install --upgrade tensorflow-gpu
#   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64



if [ ! -d "venv" ]; then
    virtualenv venv
fi

source venv/bin/activate

pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt")'
brew install coreutils



