import spacy
import pandas as pd
import streamlit as st
from spacy import displacy
 
# Load Spacy model
nlp = spacy.load('en_core_web_sm')
 
# Function to perform NER and return entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return doc, entities
 
# Function to filter and create table of PERCENT entities
def create_all_entity_table(entities):
    #percent_entities = [entity for entity in entities]
    #df = pd.DataFrame(percent_entities, columns=['Entity', 'Label'])
    df = pd.DataFrame(entities, columns=['Entity', 'Label'])
    df.drop_duplicates(inplace=True)
    return df

def create_percent_entity_table(entities):
    percent_entities = [entity for entity in entities if entity[1] == 'PERCENT']
    df = pd.DataFrame(percent_entities, columns=['Entity', 'Label'])
    df.drop_duplicates(inplace=True)
    return df

# Function to visualize named entities in the text
def visualize_entities(doc):
    options = {"ents": ["PERCENT", "ORG", "PERSON", "GPE", "DATE", "TIME"], "colors": {"PERCENT": "yellow"}}
    html = displacy.render(doc, style="ent", options=options)
    return html
 
# Streamlit App
def main():
    st.title("Named Entity Recognition (NER) Application")
   
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
   
    if uploaded_file is not None:
        # Read and process the uploaded file
        text = uploaded_file.read().decode('utf-8')
       
        # Extract entities
        doc, entities = extract_entities(text)
       
        # Create and display the entity table
        entity_table = create_all_entity_table(entities)
        st.write("Entity Table :")
        #st.dataframe(entity_table)
        st.dataframe(entity_table, width=1000, height=300)
       
        # Create two columns
        col1, col2 = st.columns(2) 

        # Download the CSV file
        csv = entity_table.to_csv(index=False).encode('utf-8')
        with col1:
            st.download_button(
            label="Download Entity Table as CSV",
            data=csv,
            file_name='entity_table.csv',
            mime='text/csv',
            )
       
        perc_entity_table = create_percent_entity_table(entities)
        csv = perc_entity_table.to_csv(index=False).encode('utf-8')
        with col2:
            st.download_button(
            label="Download only PERCENT entities",
            data=csv,
            file_name='Percent_entities_table.csv',
            mime='text/csv',
            )

        # Visualize named entities
        st.write("Named Entity Visualization:")
        html = visualize_entities(doc)
        st.markdown(html, unsafe_allow_html=True)
 
if __name__ == "__main__":
    main()
 