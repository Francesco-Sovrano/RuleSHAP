SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPT_PATH

python3.9 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -U setuptools wheel twine
pip install -r requirements.txt

# python -m spacy download en_core_web_lg
# python -m spacy download de_dep_news_lg
# python -m spacy download fr_dep_news_lg
# python -m spacy download it_core_news_lg
python -m spacy download en_core_web_md
# python.9 -m spacy download en_core_web_trf
# python -m spacy download de_core_news_md
# python -m spacy download fr_core_news_md
# python -m spacy download it_core_news_md

python -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown omw-1.4
