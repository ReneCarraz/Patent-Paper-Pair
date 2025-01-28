import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_lg")
model = SentenceTransformer("all-MiniLM-L6-v2")


# Lemmatize, tokenize and remove stopwords
def lemmatize(text):
    doc = nlp(text)
    token_list = [
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct
    ]
    return token_list


def inventor_overlap_score(inventors, authors):
    if not inventors or not authors:
        return None
    # Handle the float "NaN" case
    if type(inventors) == float or type(authors) == float:
        return None
    # Check if any entries are float
    if any([type(x) == float for x in inventors]) or any(
        [type(x) == float for x in authors]
    ):
        return None
    inventors, authors = list(filter(None, inventors)), list(filter(None, authors))
    if not inventors or not authors:
        return None
    inventor_set = set(
        map(
            lambda name: frozenset(name.lower().split(" ")),
            filter(None, inventors),
        )
    )
    author_set = set(
        map(lambda name: frozenset(name.lower().split(" ")), filter(None, authors))
    )
    if len(inventor_set) == 0:
        return None
    return len(inventor_set.intersection(author_set)) / len(inventor_set)


def doi_overlap_score(patent_dois, paper_dois):
    if not patent_dois or not paper_dois:
        return None
    # Handle the float "NaN" case
    if type(patent_dois) == float or type(paper_dois) == float:
        return None
    patent_dois, paper_dois = (
        list(filter(None, patent_dois)),
        list(filter(None, paper_dois)),
    )
    if not patent_dois or not paper_dois:
        return None
    patent_set = set(map(lambda x: x.lower(), patent_dois))
    paper_set = set(map(lambda x: x.lower(), paper_dois))
    if len(patent_set) == 0:
        return None
    return len(patent_set.intersection(paper_set)) / len(patent_set)


def semantic_similarity_score_word_overlap(
    string1,
    string2,
):
    if not string1 or not string2:
        return None

    token_list_one = lemmatize(string1)
    token_list_two = lemmatize(string2)

    string_one_set = set(map(lambda x: x.lower(), token_list_one))
    string_two_set = set(map(lambda x: x.lower(), token_list_two))
    if len(string_one_set) == 0 or len(string_two_set) == 0:
        return 0
    return len(string_one_set.intersection(string_two_set)) / min(
        len(string_one_set), len(string_two_set)
    )


def semantic_similarity_score_spacy(string_one, string_two):
    if not string_one or not string_two:
        return None
    # Try coercing to string
    try:
        string_one = str(string_one)
        string_two = str(string_two)
    except:
        return None
    doc1 = nlp(string_one)
    doc2 = nlp(string_two)
    return doc1.similarity(doc2)


def semantic_similarity_score_sbert(string_one, string_two):
    if not string_one or not string_two:
        return None
    # Try coercing to string
    try:
        string_one = str(string_one)
        string_two = str(string_two)
    except:
        return None
    embeddings = model.encode([string_one, string_two], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
    cosine_score = cosine_score.cpu().numpy()[0][0]
    if cosine_score is None:
        return 0
    return cosine_score
