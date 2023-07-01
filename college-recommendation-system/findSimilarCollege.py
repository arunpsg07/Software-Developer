#importing the necessary libraries
import pandas as pd
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_lg')


#performing normailzaiton by removing all the special characters
def normailization(text) :
    
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    
    return text


#A fucntion to perform preprocessing
def preprocessing(text) :

    #tokenization
    tokenized_word = word_tokenize(text)

    #stop word removal
    founded_stopwords = set(stopwords.words('english'))
    stopword_sentence = [token for token in tokenized_word if token not in founded_stopwords]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in stopword_sentence]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    # Join the tokens back into a single string
    preprocessed_text = " ".join(lemmatized_tokens)

    #returning the preprocessed text
    return preprocessed_text




#a function to find the similar colleges
def find_similar_college( query ):


    #reading the excel file
    data = pd.read_excel(r'E:\academic lab\IR\package\CollegeData.xlsx')


    #replacing the missing value in the table
    data['Type'] = data['Type'].replace("Public/Govt" , "Aided")
    data['Name'] = data['Name'].replace("N/A" , "NOTAVAILABLE")
    data['Location'] = data['Location'].replace("N/A" , "")
    data['State'] = data['State'].replace("N/A" , "")
    data['Rating'] = data['Rating'].replace("N/A" , "0")
    data['Total Reviews'] = data['Total Reviews'].replace("N/A" , "0")
    data['Reference Link'] = data['Reference Link'].replace("N/A" , "none")

    #Creating a new data fram
    df = pd.DataFrame( columns=['Details'] )


    #Storing the data of the college into the details of the column of the dataframe
    df['Details'] = data['Name'] + ' ' + data['Location'] + ' ' + data['State'] + ' ' + data['Type'] + ' ' + data['Total Reviews'].apply(lambda x : str(x))


    #looping all the through all the content in the details for performing normailzaiton and preprocessing
    for i in range(len(df['Details'])) :
        
        #performing normalization
        df['Details'][i] = normailization( df['Details'][i] )

        #performing all the preprocessing
        df['Details'][i] = preprocessing( df['Details'][i] )



    #Adding course offered in the college to the new datafram. Perfroming preprocessing to the course offered
    df['Details'] += ' ' + data["Courses Offered"].apply(eval).apply(lambda x: preprocessing (", ".join(x).lower().replace("." , "").replace("," , "") ))

    
    #lower casing the query
    query = query.lower()

    
    #creating the vector
    vectorizer = TfidfVectorizer(stop_words='english')
    document_vectors = vectorizer.fit_transform(df['Details'])


    #creating the query vector
    query_vectors = vectorizer.transform([query])

    
    #getting the simlarity between the query and the document vectors of the corpus
    query_similarity = cosine_similarity(query_vectors, document_vectors )
    top_indices = query_similarity.argsort()[0][::-1][:5]

    
    #finding the relevant document from the df - dataframe
    results_received = []
    for index in top_indices:
        results_received.append(df.iloc[index])

    results_received = pd.DataFrame(results_received)
    
    #getting the index of the documents from the df - dataframe
    index = []
    for i in results_received.iterrows():
        index.append(i[0])
    

    #retrieving the relevant documents from the main dataframe - data based on the index obtained.
    result = []
    for i in index:
        result.append( data.iloc[i] )

    #converting it into a dataframe
    result = pd.DataFrame(result)
    
    
    #returning the dataframe
    return result
