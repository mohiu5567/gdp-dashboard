import streamlit as st
import praw
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import wbgapi as wb
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Global Migration Pattern Analysis",
    page_icon="🌍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# Reddit API credentials input
with st.sidebar.expander("Reddit API Settings"):
    client_id = st.text_input("Client ID", type="password")
    client_secret = st.text_input("Client Secret", type="password")
    user_agent = st.text_input("User Agent", value="migration_analysis_bot")

# Analysis parameters
with st.sidebar.expander("Analysis Settings"):
    post_limit = st.slider("Number of posts to analyze", 100, 1000, 500)
    fuzzy_threshold = st.slider("Fuzzy matching threshold", 60, 100, 80)
    gdp_year = st.selectbox("GDP Data Year", options=list(range(2022, 2015, -1)))

def setup_reddit(client_id, client_secret, user_agent):
    try:
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    except Exception as e:
        st.error(f"Error setting up Reddit API: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_iwantout_posts(reddit, limit=500):
    try:
        subreddit = reddit.subreddit('IWantOut')
        posts = []
        
        for submission in subreddit.new(limit=limit):
            posts.append({
                'title': submission.title,
                'created_utc': submission.created_utc,
                'url': f"https://reddit.com{submission.permalink}",
                'score': submission.score
            })
        
        df = pd.DataFrame(posts)
        df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
        return df
    except Exception as e:
        st.error(f"Error fetching Reddit posts: {str(e)}")
        return pd.DataFrame()

def extract_countries(title):
    try:
        pattern = r'(?i).*?([a-z\s]+)\s*-+>\s*([a-z\s]+)'
        match = re.search(pattern, title)
        
        if match:
            source = match.group(1).strip()
            destination = match.group(2).strip()
            return pd.Series([source, destination])
        return pd.Series([None, None])
    except Exception as e:
        return pd.Series([None, None])

@st.cache_data(ttl=86400)
def get_gdp_data(year):
    try:
        gdp_data = wb.data.DataFrame('NY.GDP.PCAP.CD', time=year, labels=True)
        gdp_data = gdp_data.reset_index()
        gdp_data = gdp_data.rename(columns={
            'economy': 'country',
            f'YR{year}': 'gdp_per_capita'
        })
        return gdp_data[['country', 'gdp_per_capita']].dropna()
    except Exception as e:
        st.error(f"Error fetching World Bank data: {str(e)}")
        return pd.DataFrame()

def analyze_migration_patterns(reddit, post_limit, fuzzy_threshold, gdp_year):
    try:
        # Get data
        df = get_iwantout_posts(reddit, post_limit)
        if df.empty:
            return None
        
        gdp_data = get_gdp_data(gdp_year)
        if gdp_data.empty:
            return None
        
        # Extract countries
        df[['source_country', 'destination_country']] = df['title'].apply(extract_countries)
        
        # Clean countries
        valid_countries = gdp_data['country'].tolist()
        df['source_country_cleaned'] = df['source_country'].apply(
            lambda x: fuzzy_match_country(x, valid_countries, fuzzy_threshold))
        df['destination_country_cleaned'] = df['destination_country'].apply(
            lambda x: fuzzy_match_country(x, valid_countries, fuzzy_threshold))
        
        # Aggregate data
        source_counts = df['source_country_cleaned'].value_counts().reset_index()
        source_counts.columns = ['country', 'leaving_mentions']
        
        dest_counts = df['destination_country_cleaned'].value_counts().reset_index()
        dest_counts.columns = ['country', 'moving_to_mentions']
        
        # Merge data
        final_df = pd.merge(source_counts, dest_counts, on='country', how='outer')
        final_df = pd.merge(final_df, gdp_data, on='country', how='inner')
        final_df = final_df.fillna(0)
        
        return {'data': final_df, 'raw_posts': df}
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None

def create_migration_flow_chart(data):
    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=data['country'].tolist() * 2,
                color="blue"
            ),
            link=dict(
                source=list(range(len(data))),  # Source countries
                target=[x + len(data) for x in range(len(data))],  # Destination countries
                value=data['moving_to_mentions'].tolist(),
                color="rgba(0,0,255,0.2)"
            )
        )
    ])
    
    fig.update_layout(
        title_text="Migration Flow Visualization",
        font_size=12,
        height=800
    )
    
    return fig

def main():
    st.title("🌍 Global Migration Pattern Analysis")
    st.write("Analysis of migration patterns based on r/IWantOut subreddit")
    
    # Check for API credentials
    if not (client_id and client_secret):
        st.warning("Please enter your Reddit API credentials in the sidebar to begin analysis.")
        return
    
    # Initialize Reddit client
    reddit = setup_reddit(client_id, client_secret, user_agent)
    if not reddit:
        return
    
    # Run analysis
    with st.spinner('Analyzing migration patterns...'):
        result = analyze_migration_patterns(reddit, post_limit, fuzzy_threshold, gdp_year)
        
    if not result:
        return
        
    data, raw_posts = result['data'], result['raw_posts']
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🌊 Migration Flow", "📝 Raw Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Source countries plot
            fig_source = px.scatter(
                data,
                x='gdp_per_capita',
                y='leaving_mentions',
                text='country',
                title='Countries People Want to Leave vs GDP per Capita',
                log_x=True
            )
            st.plotly_chart(fig_source, use_container_width=True)
            
        with col2:
            # Destination countries plot
            fig_dest = px.scatter(
                data,
                x='gdp_per_capita',
                y='moving_to_mentions',
                text='country',
                title='Desired Destination Countries vs GDP per Capita',
                log_x=True
            )
            st.plotly_chart(fig_dest, use_container_width=True)
            
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Total Posts Analyzed", len(raw_posts))
        with col4:
            st.metric("Top Destination", 
                     data.nlargest(1, 'moving_to_mentions')['country'].iloc[0])
        with col5:
            st.metric("Top Source", 
                     data.nlargest(1, 'leaving_mentions')['country'].iloc[0])
    
    with tab2:
        # Migration flow visualization
        st.plotly_chart(create_migration_flow_chart(data), use_container_width=True)
    
    with tab3:
        # Raw data view
        st.subheader("🔍 Raw Data")
        st.dataframe(data)
        
        # Export options
        st.download_button(
            label="Download Data as CSV",
            data=data.to_csv(index=False),
            file_name=f"migration_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()


