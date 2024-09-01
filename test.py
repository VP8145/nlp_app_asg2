import streamlit as st
import spacy
#import pyarrow as pa
#import pyarrow.lib as _lib

nlp=spacy.load('en_core_web_sm')

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) + ' - ' +ent.label_+ ' - '+str(spacy.explain(ent.label_) ))

        else:
            print('no_named_entities_found.')


doc1=nlp("Apple is looking to buy U.K. startup for $1 billion")
print(type(doc1))
show_ents(doc1)


st.title('NER Application')
uploaded_file = st.file_uploader("Choose a file", type=["txt"])


#st.write(str(doc1))
                
