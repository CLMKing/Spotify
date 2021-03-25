import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

    
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score,roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

####

## Data cleaning

st.sidebar.image('splogo.png')
st.sidebar.title('Navigation Page')
st.sidebar.write('------------------------------')
navigation = st.sidebar.radio('Selection', ['Introduction', 'Client Artist','Business Objective','Data Collection and Preprocessing', 'Exploratory Data Analysis','Objective 1', 'Objective 2', 'Objective 3' ,'Business Solutions', 'Recommender Engine', 'Creators'])

if navigation == 'Introduction':
    st.title('Recommendation Engine: Exploring the Philippine Indie Music using K-means Clustering')
    st.image('splogo.png', width = 800)

if navigation == 'Client Artist':
    st.title('Spotify Song Solutions')
    st.write('------------------------------')
    opt = st.selectbox('Client Artist', ['Please select client artist','Clara Benin', 'Munimuni', 'Sleep Alley'])
    st.write('------------------------------')
    if opt == 'Clara Benin':
        st.title('Clara Benin')
        st.image('cb_image.jpg', width = 600)
        st.write('Clara Benin is the eldest daughter of former Side A member **Joey Benin**. She has three other siblings, Boey, Jaco, and Sarah who are also inclined to music')
        st.write('------------------------------')
        st.subheader('Career Info')
        st.write("In 2013, she was accepted, in the same batch with fellow artists Reese and Vica and Sitti, into the Philippines' Elements National Music Camp. Here she was mentored by Ryan Cayabyab, Joey Ayala and Noel Cabangon.")
        st.write('''In 2014, she performed for the McDonald's jingle "Hooray for Today". She then participated with Mcoy Fundales at the third Philippine Popular Music Festival to interpret the song entry “Kung Akin Ang Langit”, written by Chi Datu-Bocobo and Isaac Joseph Garcia. The song won the Spinnr People’s Choice Award.''')
        st.write('**"Human Eyes"** was recorded in three months at the Sonic Boom studio at MINT College and at Loudbox Studios. The album is self-produced, with the help of her father Joey Benin, indie music impresario Alex Lim, and studio producer/engineer Angee Rozul for mixing and mastering.')
        st.write('------------------------------')
        st.subheader('Statistics')
        st.write('Number of Listeners: 368,205')
        st.write('Popularity: 28')
        st.write('------------------------------')
        st.subheader('Popular Tracks')
        cb_df = pd.read_csv('relevant_artists_tracks_data.csv')
        cb_tracks = ['Wine', 'Parallel Universe','Tila',"Araw't Gabi", "It's Okay", 'Di na Muli (ft. Ben&Ben)', 'Closure', 'Sweet Nothings', 'Awit ng Bagong Taon']
        streams = [1221018, 4913270, 4860803, 774372, 1102711, 1069207, 2178683, 74164, 294885]
        ax = zip(cb_tracks, streams)
        clara_df_tracks = pd.DataFrame(ax, columns = ['Tracks', 'Streams']).set_index('Tracks').sort_values(by = 'Streams',ascending = False)
        st.write(clara_df_tracks)
        
    if opt == 'Munimuni':
        st.title('Munimuni')
        st.image('mm_image.jpg', width = 600)
        st.write('Munimuni are a Filipino indie folk band from Quezon City, Philippines. The band consists of AJ Jiao on lead vocals and guitar, TJ de Ocampo on lead guitar, John Owen Castro on flute and background vocals, Jolo Ferrer on bass, and Josh Tumaliuan on drums.')
        st.write('------------------------------')
        st.subheader('Career Info')
        st.write("Munimuni was formed in UP Diliman, TJ de Ocampo met AJ Jiao who was, at that time, doing a musical project with Red Calayan. They figured out that they had similar tastes in music, so they started writing and arranging songs together.A year after they started, de Ocampo was sent to Japan as an exchange student, later on adding John Owen Castro as the band's session keyboardist. Munimuni kept on doing gigs while de Ocampo was away.")
        st.write('In 2017, Munimuni released their debut EP titled "Simula". The EP featured songs like "Sa Hindi Pag-alala", "Sayo", and "Marilag"')
        
        st.write('Two years after releasing their debut EP, Munimuni finally released their debut album. The album **"Kulayan Natin"** featured songs like "Tahanan", "Oras", "Kalachuchi" and "Solomon" featuring Filipina singer Clara Benin')
        st.write('------------------------------')
        st.subheader('Statistics')
        st.write('Number of Listeners: 234,028')
        st.write('Popularity: 35')
        st.write('Number of Followers: ')
        st.write('------------------------------')
        st.subheader('Popular Tracks')
        cb_df = pd.read_csv('relevant_artists_tracks_data.csv')
        cb_tracks = ['Sa Hindi Pag-Alala', "Sa'yo", 'Bawat Piyesa',"Solomon", "Minsan", 'Tahanan', 'Simula', 'Marilag', 'Kalachuchi', 'Bakunawa']
        streams = [12163288, 7271459, 2525103, 2697205, 2179582, 4379582, 1458242, 2769467, 1614644, 845162]
        bx = zip(cb_tracks, streams)
        mm_df_tracks = pd.DataFrame(bx, columns = ['Tracks', 'Streams']).set_index('Tracks').sort_values(by = 'Streams',ascending = False)
        st.write(mm_df_tracks)
        
    if opt == 'Sleep Alley':
        st.title('Sleep Alley')
        st.image('sa_image.jpg', width = 600)
        st.write('------------------------------')
        st.subheader('Statistics')
        st.write('Number of Listeners: 25,285')
        st.write('Popularity: 21')
        st.write('------------------------------')
        st.subheader('Popular Tracks')
        cb_df = pd.read_csv('relevant_artists_tracks_data.csv')
        cb_tracks = ['Di Naging (Tayo)', "Hintay", 'Ala-Ala', "Lihim", "Makasarili", 'Dilim', 'Desperado', '88th Street', 'Uwi Bahay']
        streams = [639010, 151325, 361563, 547245, 97751, 171395, 72009, 60090, 43265, 39171]
        cx = zip(cb_tracks, streams)
        sa_df_tracks = pd.DataFrame(cx, columns = ['Tracks', 'Streams']).set_index('Tracks').sort_values(by = 'Streams',ascending = False)
        st.write(sa_df_tracks)


        
if navigation == 'Business Objective':
    st.title('Spotify Song Solutions')
    st.write('------------------------------')
    st.header('**Business Objective**')
    st.write('------------------------------')
    st.write('1. Which artists should we market together with client artists in featured playlists?')
    st.write('2. Which artist should the client artists collaborate with in their next release? ')
    st.write('3. How can client artists improve on their performance based on their audio features ')
    
        

if navigation == 'Data Collection and Preprocessing':
    st.title('Data Collection and Preprocessing')
    st.write('------------------------------')
    optt = st.selectbox('Please select information', ['Dataset Schema', 'Tools and Packages', 'Main Processes'])
    if optt == 'Dataset Schema':
        st.write('**Audio Features**')
        st.write('* **Danceability** -- describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.')
        st.write('* **Duration** -- The duration of the track in milliseconds.')
        st.write('* **Energy** -- Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.')
        st.write('* **Key** -- The key the track is in. Integers map to pitches using standard Pitch Class notation')
        st.write('* **Loudness** -- The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks.')
        st.write('* **Mode** -- Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.')
        st.write('* **Speechiness** -- Speechiness detects the presence of spoken words in a track. ')
        st.write('* **Acousticness** -- A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.')
        st.write('* **Instrumentalness** -- Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”.')
        st.write('* **Liveness** -- Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.')
        st.write('* **Valence** -- A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.')
        st.write('* **Tempo** -- The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.')
        st.write('* **Popularity** -- The popularity of the track. The value will be between 0 and 100, with 100 being the most popular.')
        st.write('* **Streams** -- Number of listeners')
    if optt == 'Tools and Packages':
        st.write('**The data scraping and data cleaning was done using the following tools**')
        st.write('* Pandas')
        st.write('* Numpy')
        st.write('* Matplotlib')
        st.write('* Seaborn')
        st.write('* Sci-kit Learn')
        st.write('* Spotipy')
    if optt == 'Main Processes':
        st.write('**Main Processes**')
        st.write('* Filtering')
        st.write('* Merging')
        st.write('* Dropping Duplicates and NaN values')
        st.write('* Groupings')
        st.write('* Labelling')

    st.image('main_tools.png', width = 1000)
    

if navigation == 'Exploratory Data Analysis':
    st.title('Spotify Song Solutions')
    st.write('------------------------------')
    st.header('Exploratory Data Analysis')
    st.write('------------------------------')
    st.subheader('What does the market looks like?')
    
    ####
    
    master_df = pd.read_csv('master_df.csv').drop(columns = 'Unnamed: 0')
    
    df_2017 = master_df[master_df['year'] == 2017]
    opm_art = df_2017[df_2017['category'] == 'opm']
    opm_art = len(opm_art['artist'].unique())
    for_art = df_2017[df_2017['category'] == 'foreign']
    for_art = len(for_art['artist'].unique())
    percent_17 = (opm_art / (opm_art + for_art))*100.0
    
    df_2018 = master_df[master_df['year'] == 2018]
    opm_art = df_2018[df_2018['category'] == 'opm']
    opm_art = len(opm_art['artist'].unique())
    for_art = df_2018[df_2018['category'] == 'foreign']
    for_art = len(for_art['artist'].unique())
    percent_18 = (opm_art / (opm_art + for_art))*100.0

    df_2019 = master_df[master_df['year'] == 2019]

    opm_art = df_2019[df_2019['category'] == 'opm']
    opm_art = len(opm_art['artist'].unique())
    for_art = df_2019[df_2019['category'] == 'foreign']
    for_art = len(for_art['artist'].unique())
    percent_19 = (opm_art / (opm_art + for_art))*100.0
    
    df_2020 = master_df[master_df['year'] == 2020]
    opm_art = df_2020[df_2020['category'] == 'opm']
    opm_art = len(opm_art['artist'].unique())
    for_art = df_2020[df_2020['category'] == 'foreign']
    for_art = len(for_art['artist'].unique())
    percent_20 = (opm_art / (opm_art + for_art))*100.0
 
    perc_lst_ = [percent_17, percent_18, percent_19, percent_20]
    year_ = [2017, 2018, 2019, 2020]
    
    fig = plt.figure(figsize = (7,7))
    sns.barplot(x = year_, y = perc_lst_, palette = 'rocket')
    plt.xlabel('Year', fontsize = 12)
    plt.ylabel('Percentage of OPM artist', fontsize = 12)
    
    ####
    
    perc_df = master_df.copy()
    perc_opm = perc_df[(perc_df['category'] == 'opm') & (perc_df['year'] == 2017)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_opm = perc_opm['streams'].sum()
    perc_for = perc_df[(perc_df['category'] != 'opm') & (perc_df['year'] == 2017)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_for = perc_for['streams'].sum()
    perc_2017 = round(total_opm/(total_opm + total_for) * 100,3)
    
    perc_opm = perc_df[(perc_df['category'] == 'opm') & (perc_df['year'] == 2018)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_opm = perc_opm['streams'].sum()
    perc_for = perc_df[(perc_df['category'] != 'opm') & (perc_df['year'] == 2018)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_for = perc_for['streams'].sum()
    perc_2018 = round(total_opm/(total_opm + total_for) * 100,3)
    
    perc_opm = perc_df[(perc_df['category'] == 'opm') & (perc_df['year'] == 2019)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_opm = perc_opm['streams'].sum()
    perc_for = perc_df[(perc_df['category'] != 'opm') & (perc_df['year'] == 2019)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_for = perc_for['streams'].sum()
    perc_2019 = round(total_opm/(total_opm + total_for) * 100,3)
    
    perc_opm = perc_df[(perc_df['category'] == 'opm') & (perc_df['year'] == 2020)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_opm = perc_opm['streams'].sum()
    perc_for = perc_df[(perc_df['category'] != 'opm') & (perc_df['year'] == 2020)].groupby('track_id')[['streams']].sum().sort_values(by='streams', ascending = False)
    total_for = perc_for['streams'].sum()
    perc_2020 = round(total_opm/(total_opm + total_for) * 100,3)
    
    yr_ = [2017, 2018, 2019, 2020]
    fig2 = plt.figure(figsize = (7,7))
    sns.barplot(x = yr_ , y = [perc_2017, perc_2018, perc_2019, perc_2020], palette = 'rocket')
    plt.ylabel('Percentage Total Streams of OPM music', fontsize = 16)
    plt.xlabel('Year', fontsize = 16)
    
    ####
    
    opt2 = st.selectbox('Please Select Needed Information', ['Default', 'Percentage of OPM artist that is in the Top 200 per year', 'Total Streams of the Top 200'])
        
    if opt2 == 'Percentage of OPM artist that is in the Top 200 per year':
        st.subheader('Percentage of OPM artist that is in the Top 200 per year')
        st.pyplot(fig)
    if opt2 == 'Total Streams of the Top 200':
        st.subheader('Total Streams of the Top 200')
        st.pyplot(fig2)
    
    st.write('------------------------------')
    
    ####
    
    df = pd.read_csv('master_df.csv')
    master_df = df[(df['category'] == 'opm')].reset_index()
    
    master_df['genres'] = master_df['genres'].apply(lambda x: x.replace("'", ''))
    master_df['genres'] = master_df['genres'].apply(lambda x: x.replace("[", ''))
    master_df['genres'] = master_df['genres'].apply(lambda x: x.replace("]", ''))
    master_df['genres'] = master_df['genres'].apply(lambda x: x.replace(" ", ''))
    master_df['genres'] = master_df['genres'].apply(lambda x: x.split(','))
    
    index = 0
    lst = []
    for item in master_df['genres']:
        lst.extend(map(lambda x: [index, x], item))
        index += 1

    genres = pd.DataFrame(lst, columns=['index', 'genres'])
    
    df_genres = pd.merge(master_df.drop('genres', axis = 1), genres, left_index = True, right_on = 'index')
    ax = df_genres[df_genres['year'] == 2017].groupby('genres')[['streams']].sum().sort_values(by = 'streams', ascending = False)
    ax = ax.rename(columns = {'streams':'streams_2017'})
    bx = df_genres[df_genres['year'] == 2018].groupby('genres')[['streams']].sum().sort_values(by = 'streams', ascending = False)
    bx = bx.rename(columns = {'streams':'streams_2018'})
    cx = df_genres[df_genres['year'] == 2019].groupby('genres')[['streams']].sum().sort_values(by = 'streams', ascending = False)
    cx = cx.rename(columns = {'streams':'streams_2019'})
    dx = df_genres[df_genres['year'] == 2020].groupby('genres')[['streams']].sum().sort_values(by = 'streams', ascending = False)
    dx = dx.rename(columns = {'streams':'streams_2020'})
    df_1 = pd.merge(ax, bx, how = 'outer', on = 'genres')
    df_2 = pd.merge(cx, dx, how = 'outer', on = 'genres')
    df_final = pd.merge(df_1, df_2, how = 'outer', on = 'genres')
    df_final = df_final.fillna(0)
    
    fig3 = plt.figure(figsize = (5,5))
    yr_ = [2017, 2018, 2019, 2020]
    sns.barplot(x = yr_, y = df_final.iloc[0])
    plt.xlabel('Year', fontsize = 16)
    plt.ylabel('OPM', fontsize = 16)
    
    fig4 = plt.figure(figsize = (5,5))
    sns.barplot(x = yr_, y = df_final.iloc[4], palette = 'rocket')
    plt.xlabel('Year', fontsize = 16)
    plt.ylabel('DancePop', fontsize = 16)
    
    fig5 = plt.figure(figsize = (5,5))
    sns.barplot(x = yr_, y = df_final.iloc[9], palette = 'rocket')
    plt.xlabel('Year', fontsize = 16)
    plt.ylabel('ViralPop', fontsize = 16)

    fig6 = plt.figure(figsize = (5,5))
    sns.barplot(x = yr_, y = df_final.iloc[10], palette = 'rocket')
    plt.xlabel('Year', fontsize = 16)
    plt.ylabel('PinoyHipHop', fontsize = 16)
    
    fig7 = plt.figure(figsize = (5,5))
    sns.barplot(x = yr_, y = df_final.iloc[11], palette = 'rocket')
    plt.xlabel('Year', fontsize = 16)
    plt.ylabel('TagalogRap', fontsize = 16)
    
    fig8 = plt.figure(figsize = (5,5))
    sns.barplot(x = yr_, y = df_final.iloc[13], palette = 'rocket')
    plt.xlabel('Year', fontsize = 16)
    plt.ylabel('AcousticPop', fontsize = 16)
    

    if st.subheader('OPM sub-genre trend of the Top 200'):
        opt3 = st.selectbox('Please Select Needed Information', ['Default', 'Viral Pop genre', 'Tagalog Rap genre', 'Pinoy Hiphop genre', 'Dance Pop genre', 'Acoustic Pop genre'])
        
        if opt3 == 'Viral Pop genre':
            st.subheader('Viral Pop genre')
            st.pyplot(fig5)
        
        if opt3 == 'Tagalog Rap genre':
            st.subheader('Tagalog Rap genre')
            st.pyplot(fig7)
        
        if opt3 == 'Pinoy Hiphop genre':
            st.subheader('Pinoy Hiphop genre')
            st.pyplot(fig6)
        
        if opt3 == 'Dance Pop genre':
            st.subheader('Dance Pop genre')
            st.pyplot(fig4)
            
        if opt3 == 'Acoustic Pop genre':
            st.subheader('Acoustic Pop genre')
            st.pyplot(fig8)

    ####
        
if navigation == 'Objective 1':  
    st.title('Spotify Song Solutions')
    st.write('------------------------------')
    st.header('Which artists should we market together with client artists in featured playlists?')
    st.write('------------------------------')
    opt4 = st.selectbox('Please select information', ['Default', 'Mean Followers For Playlists with Top 20 Artists','Most Frequently Appearing 20 Artists in Playlists with 100,000+ Followers', 'Most Popular Artists with their Streams Performance'])
    
    ####
    df_playlist = pd.read_csv('OPM_playlist_tracks_data.csv')
    df_playlist['duration'] = (df_playlist[['duration']]/60000)
    popular_playlist=df_playlist[df_playlist['total_followers'] >100000]
    highartist=popular_playlist['artist_name'].value_counts()[:20]
    
    popular_playlist=df_playlist[df_playlist['total_followers'] >100000]
    highartist=popular_playlist['artist_name'].value_counts()[:20]
    popular_artists=highartist.index.values
    
    followers_popular_artists=[]
    for ele in popular_artists:
        followers_popular_artists.append(df_playlist[df_playlist['artist_name']==ele]['total_followers'].mean())
        
    follower_artist_dic = dict(zip(popular_artists,followers_popular_artists))
    import operator
    sorted_x = sorted(follower_artist_dic.items(), key=operator.itemgetter(1))
    popular_artists=[]
    followers_popular_artists=[]
    for ele in sorted_x:
        popular_artists.append(ele[0])
        followers_popular_artists.append(ele[1]) 
    
    popular_artists.reverse()
    followers_popular_artists.reverse()
    
    sns.set_theme(style="darkgrid")

    fig9 = plt.figure(dpi = 100, figsize=(10,5))

    ax = sns.barplot(y=popular_artists, x=followers_popular_artists, ci = None, palette = 'crest')

    plt.title('Mean Followers For Playlists with Top 20 Artists', fontsize = 22)
    plt.xlabel('Followers', fontsize = 20, labelpad = 20) 
    plt.ylabel('Song Artist', fontsize = 20, rotation = 90, labelpad = 20) 
    plt.xticks(rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 10000, 
             p.get_y() + p.get_height() / 2,  '{:1.0f}'.format(width), ha = 'left', va = 'center', fontsize = 13)

    if opt4 == 'Mean Followers For Playlists with Top 20 Artists':   
        st.pyplot(fig9)

        st.write("We think that our client's inclusion on top OPM playlist will be a great predictor of streams and popularity success. Similarly, we also thought that the appearance of artists who frequently appear on popular OPM playlists would drive the success metrics of a playlist. In order to measure the success of a playlist, we used 100000+ followers as a benchmark because our results show that the playlists with this number of followers beat 92 percent of the popular OPM playlists.")
    
    ####
    
    highartist = popular_playlist['artist_name'].value_counts(ascending=True)
    highartist = pd.DataFrame(highartist)
    highartist = highartist.sort_values(by ='artist_name', ascending=False)[:20]
    
    sns.set_theme(style="darkgrid") 

    fig10 = plt.figure(dpi = 80, figsize=(10,5)) 

    ax = sns.barplot(x=highartist.index, y=highartist['artist_name'], ci = None, palette = 'crest')

    plt.title('Most Frequently Appearing 20 Artists in Playlists with 100,000+ Followers', fontsize = 18, pad = 12) 
    plt.xlabel('Artist', fontsize = 18, labelpad = 20,) 
    plt.ylabel('Frequency of Appearance', fontsize = 15, rotation = 90, labelpad = 20)
    plt.xticks(rotation=75)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)


    for p in ax.patches:
        height = p.get_height()
        ax.text(x = p.get_x()+(p.get_width()/2), y = height + 0.5, s = '{:.0f}'.format(height), ha = 'center')

    if opt4 == 'Most Frequently Appearing 20 Artists in Playlists with 100,000+ Followers':
        st.pyplot(fig10)
        st.write('Then, we get the mean followers for all playlists that include a top 20 artist. We would like to examine whether playlists including such artists are indeed more popular. From the following graph, we can see that the presence of a top 20 artist are indeed potentially good predictors for playlist success.')
    
    if opt4 == 'Most Popular Artists with their Streams Performance':
        st.image('obj3.png', width = 800)
        st.write("To further evaluate an artist's success, we checked their performance in the Top 200 Daily Charts and their daily streams throughout the period of 2017-2021. We can further conclude that Ben&Ben is a major driver of a playlist's success which gives us the idea on what playlists should we market our client.")

    if opt4 == 'Most Popular Artists with their Streams Performance':
        st.image('obj3.png', width = 600)
        st.write("To further evaluate an artist's success, we checked their performance in the Top 200 Daily Charts and their daily streams throughout the period of 2017-2021. We can further conclude that Ben&Ben is a major driver of a playlist's success which gives us the idea on what playlists should we market our client.")
        
if navigation == 'Objective 2': 
    
    st.title('Spotify Song Solutions')
    st.write('------------------------------')
    st.header('Which artist should the client artists collaborate with in their next release?')
    st.write('------------------------------')
    st.write('**KMeans Clustering** algorithm was used to classify the songs that is much closer the clients artist music using the mean audio features.')
    st.write('**Audio Features**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo')
    st.write('The final model has 3 optimal clusters and a silhouette score of 0.54. Since our client artists was put into just 2 clusters, we are going to focus on each cluster that our artist was labelled into and use it as their respective **recommender pool**')
    st.write('------------------------------')
    
    st.header('Audio Features Analysis')
    if st.checkbox('Cluster Comparison'):
        
        ####
        cluster1 = pd.read_csv('cluster1_audio_features.csv')
        cluster2 = pd.read_csv('cluster2_audio_features.csv')
        
        audio_features = ['danceability', 'energy',
       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo']
        cluster_1 = pd.read_csv('cluster1_aggregated_audio_features.csv')
        cluster_2 = pd.read_csv('cluster2_aggregated_audio_features.csv')
        relevant_labelled = pd.read_csv('relevant_labeled.csv')
        
        def removeOutlier(df,column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3-Q1

            df = df[(df[column] >= Q1 - 1.5*IQR) & 
                        (df[column] <= Q3 + 1.5*IQR)]
            return df[column]
        
        fig11,ax = plt.subplots(nrows=len(audio_features),figsize=(10,50))

        for index,feature in enumerate(audio_features):

            if feature=="mode":
                continue
            else:
                data1 = removeOutlier(cluster1,feature)
                data2 = removeOutlier(cluster2,feature)

            ax[index].set_title(feature)

            sns.histplot(data1,ax=ax[index],label="Cluster1",color="orange",kde=True)
            sns.histplot(data2,ax=ax[index],label="Cluster2",color="blue",kde=True)

            ax[index].set_xlabel("")
            ax[index].legend()

        st.pyplot(fig11)
        
        ####
        
        cluster1_anal = cluster1.drop_duplicates(subset="track_id")[audio_features+["Cluster_Labels"]]
        cluster2_anal = cluster2.drop_duplicates(subset="track_id")[audio_features+["Cluster_Labels"]]
        cluster_12 = pd.concat([cluster1_anal,cluster2_anal])
        cluster_12_anal = cluster_12.groupby("Cluster_Labels")[audio_features].mean().reset_index()
        
        from math import pi
        def make_spider( row, title, color):

            categories=list(cluster_12_anal)[1:]
            N = len(categories)

            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            ax = plt.subplot(3,3,row+1, polar=True )

            ax.set_theta_offset(pi / 3.5)
            ax.set_theta_direction(-1)

            plt.xticks(angles[:-1], categories, color='black', size=8)

            ax.set_rlabel_position(0)

            plt.yticks([-0.25, 0, 0.25, 0.5, 0.75, 1], [-0.25, 0, 0.25, 0.5,0.75, 1], color="grey", size=7) 
            plt.ylim(-0.25,1)

            values =  cluster_12_anal.loc[row].drop('Cluster_Labels').values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
            ax.fill(angles, values, color=color, alpha=0.4)

            plt.title(title, size=12, color=color, y=1.1)
        
        my_dpi=200
        fig12 = plt.figure(figsize=(2200/my_dpi, 2000/my_dpi), dpi=my_dpi)
        plt.subplots_adjust(hspace=0.5)

        # Create a color palette:
        my_palette = plt.cm.get_cmap("Set3_r", len(cluster_12_anal.index))

        for row in range(0, len(cluster_12_anal)):
            make_spider(row=row, 
                        title='Segment '+(cluster_12_anal['Cluster_Labels'][row]).astype(str), 
                        color=my_palette(row))
        st.pyplot(fig12)
        #### 
        
        st.write('------------------------------')   
        st.subheader('Analysis')
        st.write('Based on the audio features that has a more noticeable difference, **cluster 1 has a higher acousticness and energy** to it based on the number of songs compared to cluster 2.')
        st.write('Count - aggregated mean of all the audio features per artist')
        
if navigation == 'Objective 3': 
    st.title('Spotify Song Solutions')
    st.write('------------------------------')  
    st.header('How can client artists improve on their performance based on their audio features?')
    st.write('------------------------------')  
    st.write('Using multiple regression analysis and random forest regression, the audio features of a song is not indicative of its popularity. We have obtained **0.04 R squared from Random Forest** and **0.02 R squared from Linear Regression**')

    
if navigation == 'Business Solutions':
    st.write('------------------------------')  
    st.title('Spotify Song Solutions')
    st.write('------------------------------')  
    st.subheader('Client Artist Recommendation')
    
    opt5 = st.selectbox('Please select client artist', ['Default', 'Clara Benin', 'Munimuni', 'Sleep Alley'])
    st.write('------------------------------')  
    if opt5 == 'Clara Benin':
        st.header('Clara Benin')
        st.image('cb_image.jpg', width = 600)
        st.write('For Clara Benin, she was labelled on cluster 2 where the prominent audio features have a low acousticness and low acousticness. The top 3 artist was recommended.')
        c2_pool = pd.read_csv('c2_pool.csv')
        st.write(c2_pool[:3])
        if st.checkbox('Recommended artist information'):
            artist_name = ['Juris', 'Kris Lawrence', 'Jp Noche']
            monthly = [701173, 362935, 19892]
            cb_art = pd.DataFrame(zip(artist_name, monthly), columns = ['Artist Name', 'Monthly listeners']).set_index('Artist Name')
            cb_art['Similar track'] = ['Friend of Mine', 'Forever', 'Buti Na Lang Nandito Ka']
            cb_art['Artist ID'] = ['4BNWanhw4AjSXjBm9L1Jzy','11Jdq2gOTBhmfEYmFrZlue', '2SyrkGr58LI2R6jOdfJdL8']
            st.write(cb_art)
    if opt5 == 'Munimuni':
        st.header('Munimuni')
        st.image('mm_image.jpg', width = 600)
        st.write('For Munimuni, the band was labelled on cluster 1 where the prominent audio features have a high acousticness and high acousticness. The top 3 artist was recommended.')
        c1_pool = pd.read_csv('c1_pool.csv')
        st.write(c1_pool[:3])
        if st.checkbox('Recommended artist information'):
            artist_name = ['Yeng Constantion', 'Rhythm and Drip', 'ITM']
            monthly = [791505, 19486, 11614]
            mm_art = pd.DataFrame(zip(artist_name, monthly), columns = ['Artist Name', 'Monthly listeners']).set_index('Artist Name')
            mm_art['Similar track'] = ['Sana Na Lang', 'Pero', 'Luha']
            mm_art['Artist ID'] = ['0DnjaQqb436AH1idffI6CQ','5KcjmJvzOd62t9XTf27xBp', '6QmwIBi7pzFs75SnQFX6Hn']
            st.write(mm_art)
      
    if opt5 == 'Sleep Alley':
        st.header('Sleep Alley')
        st.image('sa_image.jpg', width = 600)
        st.write('For Clara Benin, she was labelled on cluster 2 where the prominent audio features have a low acousticness and low acousticness. The top 3 artist was recommended.')
        c1_pool = pd.read_csv('c1_pool.csv')
        st.write(c1_pool[:3])
        if st.checkbox('Recommended artist information'):
            artist_name = ['Yeng Constantion', 'Rhythm and Drip', 'ITM']
            monthly = [791505, 19486, 11614]
            sa_art = pd.DataFrame(zip(artist_name, monthly), columns = ['Artist Name', 'Monthly listeners']).set_index('Artist Name')
            sa_art['Similar track'] = ['Sana Na Lang', 'Pero', 'Luha']
            sa_art['Artist ID'] = ['0DnjaQqb436AH1idffI6CQ','5KcjmJvzOd62t9XTf27xBp', '6QmwIBi7pzFs75SnQFX6Hn']
            st.write(sa_art)
    
    if st.checkbox('Recommendation'):
        st.subheader('Recommendations')
        st.write("* Since playlists are one of Spotify's main product, it is worth looking what makes a playlist successful. This way, we can improve the algorithms of our generated playlist.")
        st.write("* Based on listener data, a promotional marketing (covers, tours, etc.) with most active artists like Ben&Ben and December Avenue could increase our artists' popularity and followers.")
    
    if st.checkbox('Possible Upgrades / Improvements'):
        st.subheader('Possible Upgrades / Improvements')
        st.write('* Look more other sources of data in terms of popularity or followers like **Youtube, Instagram, etc.**')
        st.write('* Explore other algorithms and approaches to tackle the problem.')
        st.write('* Use **NLP** for the lyrics (Advanced).')

if navigation == 'Recommender Engine':
    st.write('------------------------------')      
    scaler = MinMaxScaler()
    feature_cols = ['energy', 'acousticness']
    
    relevant_artists_name = ["Sleep Alley", "Munimuni","Clara Benin"]
    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
           'instrumentalness', 'liveness', 'valence', 'tempo']
    opm_all = pd.read_csv("OPM_data.csv")
    opm_all = opm_all[~opm_all["artist_name"].isin(relevant_artists_name)]
    
    opm_all['tempo'] = scaler.fit_transform(opm_all[['tempo']])
    opm_all['loudness'] = scaler.fit_transform(opm_all[['loudness']])
    
    cluster1 = pd.read_csv('cluster1_audio_features.csv')
    cluster2 = pd.read_csv('cluster2_audio_features.csv')
    
    cluster1_top10 = cluster1.groupby("track_id")["streams"].sum().sort_values(ascending=False)[:10].index
    cluster1[cluster1["track_id"].isin(cluster1_top10)]["track_id"].unique()
    
    cluster2_top10 = cluster2.groupby("track_id")["streams"].sum().sort_values(ascending=False)[:10].index
    cluster2[cluster2["track_id"].isin(cluster2_top10)]["track_id"].unique()
    
    cluster1_list = cluster1.groupby("track_id")["streams"].sum().sort_values(ascending=False).index
    cluster2_list = cluster2.groupby("track_id")["streams"].sum().sort_values(ascending=False).index
    
    cluster_1 = opm_all[opm_all['track_id'].isin(cluster1_list)]
    cluster_2 = opm_all[opm_all['track_id'].isin(cluster2_list)]
    
    #### 
    
    seed_track_data1 = opm_all[opm_all['track_name']=='Tuliro'].iloc[0]
    seed_track_data2 = opm_all[opm_all['track_name']=='Imahe'].iloc[0]
    
    ####
    
    cluster_1['cosine_dist'] = cluster_1.apply(lambda x: 1-cosine_similarity(x[feature_cols].values.reshape(1, -1),seed_track_data1[feature_cols].values.reshape(1, -1)).flatten()[0], axis=1)

    recommendation_df_1 = cluster_1[cluster_1['track_id']!=seed_track_data1['track_id']].sort_values('cosine_dist')[:5]
    recommendation_df_1 = recommendation_df_1[['track_id','track_name','artist_name','cosine_dist']]
    
    cluster_2['cosine_dist'] = cluster_2.apply(lambda x: 1-cosine_similarity(x[feature_cols].values.reshape(1, -1), seed_track_data2[feature_cols].values.reshape(1, -1)).flatten()[0], axis=1)
    
    recommendation_df_2 = cluster_2[cluster_2['track_id']!=seed_track_data2['track_id']].sort_values('cosine_dist')[:5]
    recommendation_df_2 = recommendation_df_2[['track_id','track_name','artist_name','cosine_dist']]

    st.title('Spotify Song Solutions')
    st.write('------------------------------')  
    st.subheader('Recommender Pool')
    clusters = st.radio('Please Choose which Recommender Pool', ['Cluster 1', 'Cluster 2'])
    st.write('------------------------------') 
    
    if clusters == 'Cluster 1':
        st.subheader('Cluster 1 Seed Song')
        u_input1 = st.text_input("Seed song", 'Tuliro')
        seed_track_data1 = opm_all[opm_all['track_name']== u_input1].iloc[0]
        cluster_1['cosine_dist'] = cluster_1.apply(lambda x: 1-cosine_similarity(x[feature_cols].values.reshape(1, -1),seed_track_data1[feature_cols].values.reshape(1, -1)).flatten()[0], axis=1)
        recommendation_df_1 = cluster_1[cluster_1['track_id']!=seed_track_data1['track_id']].sort_values('cosine_dist')[:5]
        recommendation_df_1 = recommendation_df_1[['track_name','artist_name']].set_index('artist_name')
        st.write(recommendation_df_1)
        if st.button('Spotify Playlist link'):
            st.write('Spotify Playlist: https://open.spotify.com/playlist/0YE0YUmUzZipGgThuEv3ti')
        
    if clusters == 'Cluster 2':
        st.subheader('Cluster 2 Seed Song')
        u_input1 = st.text_input("Seed song", 'Maybe The Night')
        seed_track_data1 = opm_all[opm_all['track_name']== u_input1].iloc[0]
        cluster_1['cosine_dist'] = cluster_1.apply(lambda x: 1-cosine_similarity(x[feature_cols].values.reshape(1, -1),seed_track_data1[feature_cols].values.reshape(1, -1)).flatten()[0], axis=1)
        recommendation_df_1 = cluster_1[cluster_1['track_id']!=seed_track_data1['track_id']].sort_values('cosine_dist')[:5]
        recommendation_df_1 = recommendation_df_1[['track_name','artist_name']].set_index('artist_name')
        st.write(recommendation_df_1)   
        if st.button('Spotify Playlist link'):
            st.write('Spotify Playlist: https://open.spotify.com/playlist/6a4HfKCGRM2QLxgmkUWL7o')


if navigation == 'Creators':  
    st.title('Spotify Song Solutions')
    st.write('------------------------------') 
    st.header('Creators')
    st.write('* Jonarie')
    st.write('* Justine')
    st.write('* Hurly')
    st.write('* King')
    st.write('------------------------------')
    st.header('Mentor')
    st.write('* Elissa')

    if st.button('Click me'):
        st.balloons()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
