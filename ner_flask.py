from flask import Flask, render_template, request, redirect, url_for
import spacy
import pandas as pd

app = Flask(__name__)

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract entities and return them in a DataFrame
def get_ents_df(doc):
    ents_data = []
    for ent in doc.ents:
        ents_data.append({
            "text": ent.text,
            "Start": ent.start_char,
            "End": ent.end_char,
            "Label": ent.label_,
            "Description": spacy.explain(ent.label_)
        })
    # Create a DataFrame from the list of dictionaries
    ents_df = pd.DataFrame(ents_data)
    return ents_df

@app.route('/', methods=['GET', 'POST'])
def index():
    entities = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith('.txt'):
            # Read file content
            text = uploaded_file.read().decode('utf-8')
            doc = nlp(text)
            ents_df = get_ents_df(doc)
            entities = ents_df.to_dict(orient='records')  # Convert DataFrame to list of dicts
        else:
            return redirect(request.url)  # Redirect if no file or wrong file type

    return render_template('index.html', entities=entities)

if __name__ == '__main__':
    app.run(debug=True)
