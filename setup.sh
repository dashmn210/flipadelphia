if [ ! -d "venv" ]; then
    virtualenv venv
fi

source venv/bin/activate

pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt")'
brew install coreutils