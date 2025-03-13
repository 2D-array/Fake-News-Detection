import streamlit as st
import re
import joblib
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Custom CSS - minimal
st.markdown("""
<style>
    .true-result {
        background-color: #1f2937; /* Dark gray */
        color: #10B981; /* Emerald green text */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 6px solid #10B981;
        font-weight: bold;
    }
    .fake-result {
        background-color: #1f2937; /* Dark gray */
        color: #EF4444; /* Red text */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 6px solid #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)



# Load models
@st.cache_resource
def load_models():
    try:
        lr = joblib.load('logistic_regression_model.pkl')
        dt = joblib.load('decision_tree_model.pkl')
        rf = joblib.load('random_forest_model.pkl')
        gb = joblib.load('gradient_boosting_model.pkl')
        vec = joblib.load('tfidf_vectorizer.pkl')
        return lr, dt, rf, gb, vec
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|<.*?>|[^\w\s]|\d|\n', ' ', text)
    return text

# Prediction function
def predict_news(text, models, vectorizer):
    # Preprocess and vectorize
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    
    # Get predictions
    predictions = []
    for model in models:
        predictions.append(model.predict(vectorized_text)[0])
    
    # Count votes
    fake_votes = predictions.count(0)
    true_votes = predictions.count(1)
    confidence = max(fake_votes, true_votes) / 4 * 100
    final_pred = 1 if true_votes > fake_votes else 0
    
    return {
        'final': final_pred,
        'confidence': confidence,
        'models': ['LR', 'DT', 'RF', 'GB'],
        'preds': predictions
    }

# Display results
def show_results(results):
    pred_class = "TRUE NEWS" if results['final'] == 1 else "FAKE NEWS"
    css_class = "true-result" if results['final'] == 1 else "fake-result"
    desc = "Likely genuine news" if results['final'] == 1 else "Potentially fake news"
    
    # Show prediction
    st.markdown(f"""
        <div class="{css_class}">
            <h3>{pred_class} (Confidence: {results['confidence']:.1f}%)</h3>
            <p>{desc}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model votes visualization
    colors = ['#EF4444' if p == 0 else '#10B981' for p in results['preds']]
    labels = ['FAKE' if p == 0 else 'TRUE' for p in results['preds']]
    
    fig = go.Figure()
    for i, model in enumerate(results['models']):
        fig.add_trace(go.Bar(
            y=[model], x=[1], orientation='h', 
            marker=dict(color=colors[i]),
            text=[labels[i]], textposition='inside'
        ))
    
    fig.update_layout(
        height=200, showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False), plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.write("Individual Model Predictions:")
    st.plotly_chart(fig, use_container_width=True)

# Sample news texts
fake_news_sample = """Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and the very dishonest fake news media. The former reality show star had just one job to do and he couldn't do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year, President Angry Pants tweeted. 2018 will be a great year for America!"""

true_news_sample = """WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a "fiscal conservative" on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS' "Face the Nation," drew a hard line on federal spending, which lawmakers are bracing to do battle over in January."""

# Main app
def main():
    # Load models
    lr, dt, rf, gb, vectorizer = load_models()
    if None in (lr, dt, rf, gb, vectorizer):
        return
    
    models = (lr, dt, rf, gb)
    
    # App header
    st.title("📰 Fake News Detector")
    st.write("Enter news text to analyze or use the test buttons below. It can give wrong result beacuse used dataset is old")
    
    # Initialize session state for text input
    if 'news_text' not in st.session_state:
        st.session_state.news_text = ""
    
    # App layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input with session state
        news_input = st.text_area("Paste news text here:", value=st.session_state.news_text, height=150)
        
        # Analyze button
        if st.button("Analyze News"):
            if news_input:
                with st.spinner("Analyzing..."):
                    results = predict_news(news_input, models, vectorizer)
                    show_results(results)
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        # Test buttons
        st.write("### Test Examples")
        
        if st.button("Test Fake News"):
            st.session_state.news_text = fake_news_sample
            st.rerun()
        
        if st.button("Test True News"):
            st.session_state.news_text = true_news_sample
            st.rerun()
        
        # About section
        st.write("### About")
        st.write("This app uses 4 ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) to detect fake news articles.")

# Run the app
if __name__ == "__main__":
    main()
